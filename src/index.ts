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

// ─── Async Job Store ─────────────────────────────────────────────────────────

interface Job {
  id: string;
  status: "pending" | "complete" | "error";
  createdAt: number;
  provider: string;
  prompt: string;
  base64?: string;
  mimeType?: string;
  wpResult?: { id: number; url: string };
  error?: string;
}

const jobs = new Map<string, Job>();

// Clean up old jobs every 10 minutes (keep for 1 hour)
setInterval(() => {
  const cutoff = Date.now() - 60 * 60 * 1000;
  for (const [id, job] of jobs) {
    if (job.createdAt < cutoff) jobs.delete(id);
  }
}, 10 * 60 * 1000);

function generateJobId(): string {
  return `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

// ─── Shared generation logic ─────────────────────────────────────────────────

interface GenerateOptions {
  provider: "openai" | "gemini";
  prompt: string;
  // OpenAI options
  model?: string;
  size?: string;
  quality?: string;
  // Gemini options
  aspect_ratio?: string;
  // Shared options
  logo?: string;
  logo_position?: "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center";
  logo_scale?: number;
  output_format?: OutputFormat;
  upload_to_wp?: boolean;
  wp_filename?: string;
}

async function runGeneration(opts: GenerateOptions): Promise<{
  base64: string;
  mimeType: string;
  wpResult?: { id: number; url: string };
}> {
  let imageBuffer: any;
  let mimeType = "image/png";

  if (opts.provider === "openai") {
    const result = await generateImageOpenAI(
      opts.prompt,
      opts.model || "gpt-image-1",
      opts.size || "1536x1024",
      opts.quality || "high"
    );
    imageBuffer = Buffer.from(result.base64, "base64");
  } else {
    const result = await generateImageGemini(opts.prompt, opts.aspect_ratio || "16:9");
    imageBuffer = Buffer.from(result.base64, "base64");
    mimeType = result.mimeType || "image/png";
  }

  // Composite logo if requested
  if (opts.logo) {
    let logoBuffer: Buffer;
    if (LOGO_URLS[opts.logo.toLowerCase()]) {
      logoBuffer = await fetchImageAsBuffer(LOGO_URLS[opts.logo.toLowerCase()]);
    } else if (opts.logo.startsWith("http")) {
      logoBuffer = await fetchImageAsBuffer(opts.logo);
    } else {
      throw new Error(`Logo '${opts.logo}' not found. Available: ${Object.keys(LOGO_URLS).join(", ") || "none"}`);
    }
    imageBuffer = await compositeLogoOnImage(
      imageBuffer, logoBuffer,
      opts.logo_position || "top-left",
      30,
      opts.logo_scale || 0.15
    );
  }

  // Convert to output format (default: avif)
  const fmt = opts.output_format || "avif";
  const fmtConfig = FORMAT_CONFIG[fmt];
  imageBuffer = await convertImageFormat(imageBuffer, fmt);
  mimeType = fmtConfig.mimeType;

  // Upload to WordPress if requested
  let wpResult: { id: number; url: string } | undefined;
  if (opts.upload_to_wp) {
    const wpUrl = process.env.WP_URL;
    const wpUser = process.env.WP_USER;
    const wpPass = process.env.WP_APP_PASSWORD;
    if (!wpUrl || !wpUser || !wpPass) {
      throw new Error("WordPress upload requires WP_URL, WP_USER, and WP_APP_PASSWORD environment variables.");
    }
    let filename = opts.wp_filename || `ai-generated-${Date.now()}${fmtConfig.ext}`;
    // Ensure filename has the correct extension
    if (!filename.endsWith(fmtConfig.ext)) {
      filename = filename.replace(/\.[^.]+$/, fmtConfig.ext);
    }
    wpResult = await uploadToWordPress(imageBuffer, filename, wpUrl, wpUser, wpPass, fmtConfig.mimeType);
  }

  return {
    base64: imageBuffer.toString("base64"),
    mimeType,
    wpResult,
  };
}



// ─── Format Configuration ────────────────────────────────────────────────────

type OutputFormat = "png" | "avif" | "webp";

const FORMAT_CONFIG: Record<OutputFormat, { mimeType: string; ext: string }> = {
  png: { mimeType: "image/png", ext: ".png" },
  avif: { mimeType: "image/avif", ext: ".avif" },
  webp: { mimeType: "image/webp", ext: ".webp" },
};

async function convertImageFormat(
  buffer: Buffer,
  format: OutputFormat,
): Promise<Buffer> {
  if (format === "avif") return sharp(buffer).avif({ quality: 80 }).toBuffer();
  if (format === "webp") return sharp(buffer).webp({ quality: 85 }).toBuffer();
  return sharp(buffer).png().toBuffer();
}

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
  wpAppPassword: string,
  contentType: string = "image/png"
): Promise<{ id: number; url: string }> {
  const auth = Buffer.from(`${wpUser}:${wpAppPassword}`).toString("base64");

  const res = await fetch(`${wpUrl}/wp-json/wp/v2/media`, {
    method: "POST",
    headers: {
      Authorization: `Basic ${auth}`,
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Content-Type": contentType,
    },
    body: imageBuffer as any,
  });

  // Log the raw response body before any parsing
  const rawBody = await res.text();
  console.error(`[WP Upload] Status: ${res.status}`);
  console.error(`[WP Upload] Content-Type sent: ${contentType}`);
  console.error(`[WP Upload] Filename: ${filename}`);
  console.error(`[WP Upload] Raw response body (first 2000 chars):\n${rawBody.slice(0, 2000)}`);

  if (!res.ok) {
    throw new Error(`WordPress upload error (${res.status}): ${rawBody}`);
  }

  // Parse JSON from the raw body (since we already consumed the stream with .text())
  const media = JSON.parse(rawBody) as { id: number; source_url: string };
  return { id: media.id, url: media.source_url };
}

// ─── MCP Server ──────────────────────────────────────────────────────────────

// Tool registration function — called once per request in stateless mode
function registerTools(server: McpServer) {

// ─── Operating README ────────────────────────────────────────────────────────

const OPERATING_README_VERSION = "2026-03-22-v2";

const OPERATING_README = `
# Image Gen MCP Server — Operating README
**Version: ${OPERATING_README_VERSION}**

You MUST call \`acknowledge_operating_readme\` with the version string above before using any other tools in this server. This ensures you understand how to use the tools correctly each session.

---

## Overview
This MCP server generates AI images via OpenAI (GPT-Image-1) and Google Gemini (2.5 Flash Image), with built-in logo compositing (Sharp) and WordPress upload support. It is owned and operated by Shane Martin for use across Tradeify, Tradeify Crypto, and BestProps.

---

## Tools Available

### Image Generation (sync — use for quick/medium quality)
- **imagegen_openai** — Generate via OpenAI GPT-Image-1 or DALL-E 3. Best quality but slower. Use \`quality: "medium"\` to stay under the 45s MCP timeout. Supports logo compositing and WP upload.
- **imagegen_gemini** — Generate via Google Gemini 2.5 Flash Image. Faster than OpenAI, excellent at rendering text in images. Supports logo compositing and WP upload. Use \`aspect_ratio: "16:9"\` for blog headers.

### Image Generation (async — use for high quality)
- **imagegen_start** — Kicks off generation in the background. Returns a job ID immediately. Use this when you need \`quality: "high"\` with OpenAI, or whenever you want to avoid timeout risk. Accepts all the same params as the sync tools (provider, prompt, logo, quality, upload_to_wp, etc.).
- **imagegen_result** — Poll for the result of an async job by job_id. Returns "pending" if still processing, the completed image if done, or an error message. Wait ~15-30 seconds between polls. Jobs expire after 1 hour.

### Utility
- **imagegen_composite_logo** — Add a logo to any existing image URL. Useful for branding images from other sources.
- **imagegen_list_logos** — List all pre-configured logo keys and their URLs.
- **imagegen_status** — Check which providers (OpenAI, Gemini, WordPress) are configured and available.

---

## Pre-configured Logos
Logos are set via environment variables (LOGO_*). Current logos:
- \`tradeify\` — Tradeify futures prop firm logo
- \`bestprops\` — BestProps.com prop firm directory logo

Pass the key as the \`logo\` parameter (e.g., \`logo: "tradeify"\`). You can also pass a full URL to any PNG image.

---

## Logo Compositing Options
- \`logo_position\`: "top-left" (default), "top-right", "bottom-left", "bottom-right", "center"
- \`logo_scale\`: 0.05 to 0.5 (default 0.15 = 15% of image width). Use 0.12 for subtle branding, 0.2 for prominent.

---

## WordPress Upload
Set \`upload_to_wp: true\` on any generation tool to upload the final image directly to the BestProps WordPress media library. Optionally set \`wp_filename\` for a descriptive filename (e.g., "mobile-dom-alerts-header.png"). The response will include the WordPress URL and media ID.

WordPress is configured for BestProps.com (https://bestprops.com). For Tradeify (Webflow) and Tradeify Crypto (Webflow), images need to be uploaded separately via the Webflow MCP or manually.

---

## Recommended Workflow

### For blog headers (most common use case):
1. Use \`imagegen_gemini\` with \`aspect_ratio: "16:9"\`, the appropriate \`logo\` key, and \`upload_to_wp: true\` if it's a BestProps article.
2. If the user wants OpenAI high quality, use \`imagegen_start\` with \`provider: "openai"\` and \`quality: "high"\`, then poll with \`imagegen_result\`.
3. Always include the blog title in the prompt if the user wants text in the image — Gemini handles text rendering much better than OpenAI.

### For quick iterations:
Use \`imagegen_openai\` with \`quality: "medium"\` — it stays under the 45s timeout and still looks good.

### Provider comparison:
- **Gemini**: Faster, better at text in images, $0.039/image, 16:9 aspect ratio native
- **OpenAI GPT-Image-1**: Highest overall quality, slower, better at photorealism, needs async for high quality

---

## Important Notes
- The 45-second MCP timeout is a Claude.ai platform limit. It cannot be changed. Use \`imagegen_start\`/\`imagegen_result\` for anything that might exceed it.
- Logo images should ideally have transparent backgrounds for clean compositing.
- The base64 image returned by the tools may not render inline in Claude.ai chat. Always use \`upload_to_wp: true\` or tell the user to check the WordPress media library for the final result.
- When generating blog headers, default to 16:9 aspect ratio (Gemini) or 1536x1024 (OpenAI).

---

## Output Format (AVIF by default)
All generation tools support an \`output_format\` parameter: "avif" (default), "webp", or "png".
- **AVIF**: 50-80% smaller than PNG, excellent quality. Default for all generation. Best for page speed.
- **WebP**: Widely supported, ~30-50% smaller than PNG. Good fallback if AVIF causes issues.
- **PNG**: Lossless, largest file size. Use only when lossless quality is required.

The format conversion happens via Sharp after logo compositing. WordPress filenames and Content-Type headers are automatically updated to match the output format.
`.trim();

let readmeAcknowledged = false;

server.tool(
  "imagegen_get_readme",
  "Return the operating instructions for the Image Gen MCP server. MUST be called and acknowledged before using any other tools in this server.",
  {},
  async () => {
    return {
      content: [{ type: "text", text: OPERATING_README }],
    };
  }
);

server.tool(
  "imagegen_acknowledge_readme",
  "Acknowledge that you have read and understood the operating readme. Must pass the current version string from the readme.",
  {
    version: z.string().describe("The version string from the operating readme (e.g., '2026-03-22-v1')."),
  },
  async ({ version }) => {
    if (version !== OPERATING_README_VERSION) {
      return {
        isError: true,
        content: [{
          type: "text",
          text: `Version mismatch. Expected '${OPERATING_README_VERSION}', got '${version}'. Please call get_operating_readme again to get the latest version.`,
        }],
      };
    }
    readmeAcknowledged = true;
    return {
      content: [{
        type: "text",
        text: `Operating readme acknowledged (version ${version}). All tools are now available. Ready to generate images.`,
      }],
    };
  }
);

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
    wp_filename: z.string().optional().describe("Filename for WordPress upload (e.g. 'mobile-dom-alerts-header.avif')."),
    output_format: z.enum(["png", "avif", "webp"]).default("avif").describe("Output image format. AVIF is smallest (50-80% smaller than PNG), WebP is widely supported, PNG is lossless. Default: avif."),
  },
  async ({ prompt, model, size, quality, logo, logo_position, logo_scale, upload_to_wp, wp_filename, output_format }) => {
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

      // Convert to output format
      const fmt = output_format || "avif";
      const fmtConfig = FORMAT_CONFIG[fmt];
      imageBuffer = await convertImageFormat(imageBuffer, fmt);

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
        let finalFilename = filename.endsWith(fmtConfig.ext) ? filename : filename.replace(/\.[^.]+$/, fmtConfig.ext);
        wpResult = await uploadToWordPress(imageBuffer, finalFilename, wpUrl, wpUser, wpPass, fmtConfig.mimeType);
      }

      const base64Final = imageBuffer.toString("base64");

      const textParts: string[] = [
        `Image generated successfully via OpenAI ${model} (format: ${fmt}).`,
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
          { type: "image", data: base64Final, mimeType: fmtConfig.mimeType },
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
    output_format: z.enum(["png", "avif", "webp"]).default("avif").describe("Output image format. AVIF is smallest, WebP is widely supported, PNG is lossless. Default: avif."),
  },
  async ({ prompt, aspect_ratio, logo, logo_position, logo_scale, upload_to_wp, wp_filename, output_format }) => {
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

      // Convert to output format
      const fmt = output_format || "avif";
      const fmtConfig = FORMAT_CONFIG[fmt];
      imageBuffer = await convertImageFormat(imageBuffer, fmt);

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
        let filename = wp_filename || `ai-generated-${Date.now()}${fmtConfig.ext}`;
        if (!filename.endsWith(fmtConfig.ext)) {
          filename = filename.replace(/\.[^.]+$/, fmtConfig.ext);
        }
        wpResult = await uploadToWordPress(imageBuffer, filename, wpUrl, wpUser, wpPass, fmtConfig.mimeType);
      }

      const base64Final = imageBuffer.toString("base64");

      const textParts: string[] = [
        `Image generated successfully via Google Gemini (format: ${fmt}).`,
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
          { type: "image", data: base64Final, mimeType: fmtConfig.mimeType },
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

// Tool: Async start - kicks off generation and returns immediately with job ID
server.tool(
  "imagegen_start",
  "Start an async image generation job. Returns a job ID immediately — use imagegen_result to poll for the completed image. Use this for high-quality generation that might exceed timeout limits.",
  {
    provider: z.enum(["openai", "gemini"]).describe("Which AI provider to use. 'gemini' is faster and better at text in images. 'openai' (gpt-image-1) has highest quality but slower."),
    prompt: z.string().describe("Detailed image generation prompt."),
    model: z.enum(["gpt-image-1", "dall-e-3"]).default("gpt-image-1").optional().describe("OpenAI model (only used when provider is 'openai')."),
    size: z.enum(["1024x1024", "1536x1024", "1024x1536", "auto"]).default("1536x1024").optional().describe("Image size (only used when provider is 'openai')."),
    quality: z.enum(["low", "medium", "high", "auto"]).default("high").optional().describe("Image quality (only used when provider is 'openai'). High quality is safe here since there's no timeout."),
    aspect_ratio: z.enum(["1:1", "16:9", "9:16", "4:3", "3:4"]).default("16:9").optional().describe("Aspect ratio (only used when provider is 'gemini')."),
    logo: z.string().optional().describe("Logo key (e.g. 'tradeify') or URL to a logo PNG."),
    logo_position: z.enum(["top-left", "top-right", "bottom-left", "bottom-right", "center"]).default("top-left").optional(),
    logo_scale: z.number().min(0.05).max(0.5).default(0.15).optional(),
    upload_to_wp: z.boolean().default(false).optional().describe("Upload to WordPress when complete."),
    wp_filename: z.string().optional().describe("Filename for WordPress upload."),
    output_format: z.enum(["png", "avif", "webp"]).default("avif").optional().describe("Output image format. Default: avif."),
  },
  async ({ provider, prompt, model, size, quality, aspect_ratio, logo, logo_position, logo_scale, upload_to_wp, wp_filename, output_format }) => {
    const jobId = generateJobId();
    const job: Job = {
      id: jobId,
      status: "pending",
      createdAt: Date.now(),
      provider,
      prompt,
    };
    jobs.set(jobId, job);

    // Fire and forget — run generation in background
    runGeneration({
      provider,
      prompt,
      model: model || undefined,
      size: size || undefined,
      quality: quality || undefined,
      aspect_ratio: aspect_ratio || undefined,
      logo: logo || undefined,
      logo_position: (logo_position as any) || undefined,
      logo_scale: logo_scale || undefined,
      output_format: (output_format as OutputFormat) || undefined,
      upload_to_wp: upload_to_wp || false,
      wp_filename: wp_filename || undefined,
    }).then((result) => {
      job.status = "complete";
      job.base64 = result.base64;
      job.mimeType = result.mimeType;
      job.wpResult = result.wpResult;
    }).catch((err) => {
      job.status = "error";
      job.error = err instanceof Error ? err.message : String(err);
    });

    return {
      content: [{
        type: "text",
        text: `Job started: ${jobId}\nProvider: ${provider}\nStatus: pending\n\nUse imagegen_result with this job_id to check when the image is ready. It typically takes 15-60 seconds depending on provider and quality.`,
      }],
    };
  }
);

// Tool: Poll for async job result
server.tool(
  "imagegen_result",
  "Check the result of an async image generation job started with imagegen_start. Returns the image if complete, or status if still pending.",
  {
    job_id: z.string().describe("The job ID returned by imagegen_start."),
  },
  async ({ job_id }) => {
    const job = jobs.get(job_id);
    if (!job) {
      return {
        isError: true,
        content: [{ type: "text", text: `Job '${job_id}' not found. It may have expired (jobs are kept for 1 hour).` }],
      };
    }

    if (job.status === "pending") {
      const elapsed = Math.round((Date.now() - job.createdAt) / 1000);
      return {
        content: [{
          type: "text",
          text: `Job ${job_id} is still processing (${elapsed}s elapsed).\nProvider: ${job.provider}\n\nTry again in 10-15 seconds.`,
        }],
      };
    }

    if (job.status === "error") {
      return {
        isError: true,
        content: [{ type: "text", text: `Job ${job_id} failed: ${job.error}` }],
      };
    }

    // Status is "complete"
    const textParts: string[] = [
      `Job ${job_id} complete!`,
      `Provider: ${job.provider}`,
    ];
    if (job.wpResult) {
      textParts.push(`Uploaded to WordPress: ${job.wpResult.url} (media ID: ${job.wpResult.id})`);
    }

    const content: Array<{ type: string; text?: string; data?: string; mimeType?: string }> = [
      { type: "text", text: textParts.join("\n") },
    ];

    if (job.base64) {
      content.push({
        type: "image",
        data: job.base64,
        mimeType: job.mimeType || "image/png",
      });
    }

    // Clean up after retrieval
    jobs.delete(job_id);

    return { content: content as any };
  }
);

// ─── Express Server with Streamable HTTP Transport ───────────────────────────
} // end registerTools

const app = express();
app.use(express.json());

// Factory function to create a new MCP server with all tools registered
function createMcpServer(): McpServer {
  const srv = new McpServer({
    name: "image-gen-mcp-server",
    version: "1.0.0",
  });

  // Re-register all tools on the new server instance
  registerTools(srv);
  return srv;
}

app.all("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined, // stateless
  });

  const srv = createMcpServer();

  res.on("close", () => {
    transport.close();
    srv.close();
  });

  await srv.connect(transport);
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
