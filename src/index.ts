import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";
import sharp from "sharp";

// ─── Configuration ───────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT || "3000", 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";

// Pre-configured logo URLs (add your brand logos here as env vars)
// e.g. LOGO_TRADEIFY=https://cdn.prod.website-files.com/...logo.png
const LOGO_URLS: Record<string, string> = {};
for (const [key, value] of Object.entries(process.env)) {
  if (key.startsWith("LOGO_") && value) {
    const name = key.replace("LOGO_", "").toLowerCase();
    LOGO_URLS[name] = value;
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function fetchImageAsBuffer(url: string): Promise<Buffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch image from ${url}: ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function compositeLogoOnImage(
  imageBuffer: Buffer,
  logoBuffer: Buffer,
  position: "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center" = "top-left",
  padding: number = 30,
  logoScale: number = 0.15
): Promise<Buffer> {
  const image = sharp(imageBuffer);
  const metadata = await image.metadata();
  const imgW = metadata.width || 1200;
  const imgH = metadata.height || 630;

  // Scale logo relative to image width
  const targetLogoW = Math.round(imgW * logoScale);
  const logo = await sharp(logoBuffer)
    .resize({ width: targetLogoW, withoutEnlargement: false })
    .toBuffer();

  const logoMeta = await sharp(logo).metadata();
  const logoW = logoMeta.width || targetLogoW;
  const logoH = logoMeta.height || 50;

  let left: number, top: number;
  switch (position) {
    case "top-left":
      left = padding;
      top = padding;
      break;
    case "top-right":
      left = imgW - logoW - padding;
      top = padding;
      break;
    case "bottom-left":
      left = padding;
      top = imgH - logoH - padding;
      break;
    case "bottom-right":
      left = imgW - logoW - padding;
      top = imgH - logoH - padding;
      break;
    case "center":
      left = Math.round((imgW - logoW) / 2);
      top = Math.round((imgH - logoH) / 2);
      break;
  }

  return image
    .composite([{ input: logo, left, top }])
    .png()
    .toBuffer();
}

// ─── OpenAI Image Generation ─────────────────────────────────────────────────

async function generateImageOpenAI(
  prompt: string,
  model: string = "gpt-image-1",
  size: string = "1536x1024",
  quality: string = "high"
): Promise<{ base64: string; revisedPrompt?: string }> {
  if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY not configured");

  const body: Record<string, unknown> = {
    model,
    prompt,
    n: 1,
    size,
    quality,
  };

  // gpt-image-1 returns base64 by default
  if (model === "dall-e-3") {
    body.response_format = "b64_json";
  }

  const res = await fetch("https://api.openai.com/v1/images/generations", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error (${res.status}): ${err}`);
  }

  const data = await res.json() as {
    data: Array<{ b64_json?: string; url?: string; revised_prompt?: string }>;
  };

  const img = data.data[0];
  if (!img) throw new Error("No image returned from OpenAI");

  let base64 = img.b64_json || "";
  if (!base64 && img.url) {
    const buf = await fetchImageAsBuffer(img.url);
    base64 = buf.toString("base64");
  }

  return { base64, revisedPrompt: img.revised_prompt };
}

// ─── Gemini Image Generation ─────────────────────────────────────────────────

async function generateImageGemini(
  prompt: string,
  aspectRatio: string = "16:9"
): Promise<{ base64: string; mimeType: string }> {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY not configured");

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=${GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          {
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          responseModalities: ["TEXT", "IMAGE"],
          imageConfig: {
            aspectRatio: aspectRatio,
          },
        },
      }),
    }
  );

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini API error (${res.status}): ${err}`);
  }

  const data = await res.json() as {
    candidates: Array<{
      content: {
        parts: Array<{ inlineData?: { mimeType: string; data: string }; text?: string }>;
      };
    }>;
  };

  const parts = data.candidates?.[0]?.content?.parts || [];
  const imagePart = parts.find((p) => p.inlineData);
  if (!imagePart?.inlineData) throw new Error("No image returned from Gemini");

  return {
    base64: imagePart.inlineData.data,
    mimeType: imagePart.inlineData.mimeType,
  };
}

// ─── Upload to WordPress ─────────────────────────────────────────────────────

async function uploadToWordPress(
  imageBuffer: Buffer,
  filename: string,
  wpUrl: string,
  wpUser: string,
  wpAppPassword: string
): Promise<{ id: number; url: string }> {
  const auth = Buffer.from(`${wpUser}:${wpAppPassword}`).toString("base64");

  const res = await fetch(`${wpUrl}/wp-json/wp/v2/media`, {
    method: "POST",
    headers: {
      Authorization: `Basic ${auth}`,
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Content-Type": "image/png",
    },
    body: imageBuffer as any,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`WordPress upload error (${res.status}): ${err}`);
  }

  const media = await res.json() as { id: number; source_url: string };
  return { id: media.id, url: media.source_url };
}

// ─── MCP Server ──────────────────────────────────────────────────────────────

const server = new McpServer({
  name: "image-gen-mcp-server",
  version: "1.0.0",
});

// Tool: Generate image with OpenAI
server.tool(
  "imagegen_openai",
  "Generate an image using OpenAI's GPT-Image-1 or DALL-E 3 models. Returns base64 image data. Supports logo compositing and optional WordPress upload.",
  {
    prompt: z.string().describe("Detailed image generation prompt. Include style, lighting, composition details for best results."),
    model: z.enum(["gpt-image-1", "dall-e-3"]).default("gpt-image-1").describe("OpenAI model to use. gpt-image-1 is newest with best quality."),
    size: z.enum(["1024x1024", "1536x1024", "1024x1536", "auto"]).default("1536x1024").describe("Image dimensions. 1536x1024 is landscape, good for blog headers."),
    quality: z.enum(["low", "medium", "high", "auto"]).default("high").describe("Image quality. Higher = slower but better."),
    logo: z.string().optional().describe("Logo key (e.g. 'tradeify') from pre-configured logos, OR a URL to a logo PNG image. Logo will be composited onto the generated image."),
    logo_position: z.enum(["top-left", "top-right", "bottom-left", "bottom-right", "center"]).default("top-left").describe("Where to place the logo on the image."),
    logo_scale: z.number().min(0.05).max(0.5).default(0.15).describe("Logo size as fraction of image width (0.15 = 15% of image width)."),
    upload_to_wp: z.boolean().default(false).describe("Upload the final image to WordPress and return the media URL."),
    wp_filename: z.string().optional().describe("Filename for WordPress upload (e.g. 'mobile-dom-alerts-header.png')."),
  },
  async ({ prompt, model, size, quality, logo, logo_position, logo_scale, upload_to_wp, wp_filename }) => {
    try {
      // Generate the image
      const result = await generateImageOpenAI(prompt, model, size, quality);
      let imageBuffer: any = Buffer.from(result.base64, "base64");

      // Composite logo if requested
      if (logo) {
        let logoBuffer: Buffer;
        if (LOGO_URLS[logo.toLowerCase()]) {
          logoBuffer = await fetchImageAsBuffer(LOGO_URLS[logo.toLowerCase()]);
        } else if (logo.startsWith("http")) {
          logoBuffer = await fetchImageAsBuffer(logo);
        } else {
          return {
            isError: true,
            content: [{ type: "text", text: `Logo '${logo}' not found. Available logos: ${Object.keys(LOGO_URLS).join(", ") || "none configured"}. You can also pass a URL to a logo PNG.` }],
          };
        }
        imageBuffer = await compositeLogoOnImage(imageBuffer, logoBuffer, logo_position, 30, logo_scale);
      }

      // Upload to WordPress if requested
      let wpResult: { id: number; url: string } | null = null;
      if (upload_to_wp) {
        const wpUrl = process.env.WP_URL;
        const wpUser = process.env.WP_USER;
        const wpPass = process.env.WP_APP_PASSWORD;
        if (!wpUrl || !wpUser || !wpPass) {
          return {
            isError: true,
            content: [{ type: "text", text: "WordPress upload requires WP_URL, WP_USER, and WP_APP_PASSWORD environment variables." }],
          };
        }
        const filename = wp_filename || `ai-generated-${Date.now()}.png`;
        wpResult = await uploadToWordPress(imageBuffer, filename, wpUrl, wpUser, wpPass);
      }

      const base64Final = imageBuffer.toString("base64");

      const textParts: string[] = [
        `Image generated successfully via OpenAI ${model}.`,
      ];
      if (result.revisedPrompt) {
        textParts.push(`Revised prompt: ${result.revisedPrompt}`);
      }
      if (logo) {
        textParts.push(`Logo composited at ${logo_position} (scale: ${logo_scale}).`);
      }
      if (wpResult) {
        textParts.push(`Uploaded to WordPress: ${wpResult.url} (media ID: ${wpResult.id})`);
      }

      return {
        content: [
          { type: "text", text: textParts.join("\n") },
          { type: "image", data: base64Final, mimeType: "image/png" },
        ],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
      };
    }
  }
);

// Tool: Generate image with Gemini
server.tool(
  "imagegen_gemini",
  "Generate an image using Google Gemini's image generation capabilities. Returns base64 image data. Supports logo compositing and optional WordPress upload.",
  {
    prompt: z.string().describe("Detailed image generation prompt. Gemini excels at text rendering in images and photorealistic output."),
    aspect_ratio: z.enum(["1:1", "16:9", "9:16", "4:3", "3:4"]).default("16:9").describe("Aspect ratio. 16:9 is ideal for blog headers."),
    logo: z.string().optional().describe("Logo key (e.g. 'tradeify') from pre-configured logos, OR a URL to a logo PNG image."),
    logo_position: z.enum(["top-left", "top-right", "bottom-left", "bottom-right", "center"]).default("top-left").describe("Where to place the logo on the image."),
    logo_scale: z.number().min(0.05).max(0.5).default(0.15).describe("Logo size as fraction of image width."),
    upload_to_wp: z.boolean().default(false).describe("Upload the final image to WordPress and return the media URL."),
    wp_filename: z.string().optional().describe("Filename for WordPress upload."),
  },
  async ({ prompt, aspect_ratio, logo, logo_position, logo_scale, upload_to_wp, wp_filename }) => {
    try {
      const result = await generateImageGemini(prompt, aspect_ratio);
      let imageBuffer: any = Buffer.from(result.base64, "base64");

      // Composite logo if requested
      if (logo) {
        let logoBuffer: Buffer;
        if (LOGO_URLS[logo.toLowerCase()]) {
          logoBuffer = await fetchImageAsBuffer(LOGO_URLS[logo.toLowerCase()]);
        } else if (logo.startsWith("http")) {
          logoBuffer = await fetchImageAsBuffer(logo);
        } else {
          return {
            isError: true,
            content: [{ type: "text", text: `Logo '${logo}' not found. Available logos: ${Object.keys(LOGO_URLS).join(", ") || "none configured"}. You can also pass a URL to a logo PNG.` }],
          };
        }
        imageBuffer = await compositeLogoOnImage(imageBuffer, logoBuffer, logo_position, 30, logo_scale);
      }

      // Upload to WordPress if requested
      let wpResult: { id: number; url: string } | null = null;
      if (upload_to_wp) {
        const wpUrl = process.env.WP_URL;
        const wpUser = process.env.WP_USER;
        const wpPass = process.env.WP_APP_PASSWORD;
        if (!wpUrl || !wpUser || !wpPass) {
          return {
            isError: true,
            content: [{ type: "text", text: "WordPress upload requires WP_URL, WP_USER, and WP_APP_PASSWORD environment variables." }],
          };
        }
        const filename = wp_filename || `ai-generated-${Date.now()}.png`;
        wpResult = await uploadToWordPress(imageBuffer, filename, wpUrl, wpUser, wpPass);
      }

      const base64Final = imageBuffer.toString("base64");

      const textParts: string[] = [
        "Image generated successfully via Google Gemini.",
      ];
      if (logo) {
        textParts.push(`Logo composited at ${logo_position} (scale: ${logo_scale}).`);
      }
      if (wpResult) {
        textParts.push(`Uploaded to WordPress: ${wpResult.url} (media ID: ${wpResult.id})`);
      }

      return {
        content: [
          { type: "text", text: textParts.join("\n") },
          { type: "image", data: base64Final, mimeType: result.mimeType || "image/png" },
        ],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
      };
    }
  }
);

// Tool: Composite logo onto an existing image
server.tool(
  "imagegen_composite_logo",
  "Composite a logo onto an existing image from a URL. Useful for adding branding to any image.",
  {
    image_url: z.string().url().describe("URL of the base image to add the logo to."),
    logo: z.string().describe("Logo key (e.g. 'tradeify') from pre-configured logos, OR a URL to a logo PNG image."),
    position: z.enum(["top-left", "top-right", "bottom-left", "bottom-right", "center"]).default("top-left"),
    logo_scale: z.number().min(0.05).max(0.5).default(0.15),
    upload_to_wp: z.boolean().default(false),
    wp_filename: z.string().optional(),
  },
  async ({ image_url, logo, position, logo_scale, upload_to_wp, wp_filename }) => {
    try {
      const imageBuffer = await fetchImageAsBuffer(image_url);

      let logoBuffer: Buffer;
      if (LOGO_URLS[logo.toLowerCase()]) {
        logoBuffer = await fetchImageAsBuffer(LOGO_URLS[logo.toLowerCase()]);
      } else if (logo.startsWith("http")) {
        logoBuffer = await fetchImageAsBuffer(logo);
      } else {
        return {
          isError: true,
          content: [{ type: "text", text: `Logo '${logo}' not found. Available: ${Object.keys(LOGO_URLS).join(", ") || "none"}` }],
        };
      }

      const result = await compositeLogoOnImage(imageBuffer, logoBuffer, position, 30, logo_scale);

      let wpResult: { id: number; url: string } | null = null;
      if (upload_to_wp) {
        const wpUrl = process.env.WP_URL;
        const wpUser = process.env.WP_USER;
        const wpPass = process.env.WP_APP_PASSWORD;
        if (!wpUrl || !wpUser || !wpPass) {
          return {
            isError: true,
            content: [{ type: "text", text: "WordPress upload requires WP_URL, WP_USER, and WP_APP_PASSWORD environment variables." }],
          };
        }
        const filename = wp_filename || `logo-composite-${Date.now()}.png`;
        wpResult = await uploadToWordPress(result, filename, wpUrl, wpUser, wpPass);
      }

      const textParts = [`Logo composited at ${position} (scale: ${logo_scale}).`];
      if (wpResult) textParts.push(`Uploaded to WordPress: ${wpResult.url} (media ID: ${wpResult.id})`);

      return {
        content: [
          { type: "text", text: textParts.join("\n") },
          { type: "image", data: result.toString("base64"), mimeType: "image/png" },
        ],
      };
    } catch (error) {
      return {
        isError: true,
        content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
      };
    }
  }
);

// Tool: List available logos
server.tool(
  "imagegen_list_logos",
  "List all pre-configured logo keys available for compositing. Logos are set via LOGO_* environment variables.",
  {},
  async () => {
    const logos = Object.entries(LOGO_URLS);
    if (logos.length === 0) {
      return {
        content: [{ type: "text", text: "No logos configured. Set environment variables like LOGO_TRADEIFY=https://... to add logos." }],
      };
    }
    const list = logos.map(([key, url]) => `- **${key}**: ${url}`).join("\n");
    return {
      content: [{ type: "text", text: `Available logos:\n${list}` }],
    };
  }
);

// Tool: Check which providers are configured
server.tool(
  "imagegen_status",
  "Check which image generation providers are configured and available.",
  {},
  async () => {
    const status: string[] = [];
    status.push(`OpenAI: ${OPENAI_API_KEY ? "✅ Configured" : "❌ Not configured (set OPENAI_API_KEY)"}`);
    status.push(`Gemini: ${GEMINI_API_KEY ? "✅ Configured" : "❌ Not configured (set GEMINI_API_KEY)"}`);
    status.push(`WordPress: ${process.env.WP_URL ? "✅ Configured" : "❌ Not configured (set WP_URL, WP_USER, WP_APP_PASSWORD)"}`);
    status.push(`Logos: ${Object.keys(LOGO_URLS).length > 0 ? Object.keys(LOGO_URLS).join(", ") : "None configured"}`);
    return {
      content: [{ type: "text", text: status.join("\n") }],
    };
  }
);

// ─── Express Server with Streamable HTTP Transport ───────────────────────────

const app = express();
app.use(express.json());

app.all("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined, // stateless
  });

  res.on("close", () => {
    transport.close();
    server.close();
  });

  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

// Health check
app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    providers: {
      openai: !!OPENAI_API_KEY,
      gemini: !!GEMINI_API_KEY,
      wordpress: !!process.env.WP_URL,
    },
    logos: Object.keys(LOGO_URLS),
  });
});

app.listen(PORT, () => {
  console.error(`image-gen-mcp-server running on port ${PORT}`);
  console.error(`MCP endpoint: http://localhost:${PORT}/mcp`);
  console.error(`Health check: http://localhost:${PORT}/health`);
  console.error(`Providers: OpenAI=${!!OPENAI_API_KEY}, Gemini=${!!GEMINI_API_KEY}`);
  console.error(`Logos: ${Object.keys(LOGO_URLS).join(", ") || "none"}`);
});
