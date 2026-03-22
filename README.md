# Image Gen MCP Server

MCP server for AI image generation via **OpenAI (GPT-Image-1 / DALL-E 3)** and **Google Gemini Imagen**, with built-in logo compositing and WordPress upload support.

Designed to work with Claude.ai as a connected MCP integration, enabling prompt-optimized image generation directly in conversations — no manual prompt tuning needed.

## Features

- **Dual Provider Support**: Generate images via OpenAI or Gemini from the same server
- **Logo Compositing**: Automatically overlay brand logos at configurable positions and sizes using Sharp
- **WordPress Upload**: Optionally upload generated images directly to WordPress as media
- **Pre-configured Logos**: Set logos as env vars once, reference them by key (e.g. `logo: "tradeify"`)
- **Status Check**: Verify which providers and features are configured

## Tools

| Tool | Description |
|------|-------------|
| `imagegen_openai` | Generate image via OpenAI GPT-Image-1 or DALL-E 3 |
| `imagegen_gemini` | Generate image via Google Gemini |
| `imagegen_composite_logo` | Add a logo to any existing image URL |
| `imagegen_list_logos` | List pre-configured logo keys |
| `imagegen_status` | Check provider configuration status |

## Environment Variables

### Required (at least one provider)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-Image-1 / DALL-E 3 |
| `GEMINI_API_KEY` | Google Gemini API key for Imagen |

### Optional - WordPress Upload

| Variable | Description |
|----------|-------------|
| `WP_URL` | WordPress site URL (e.g. `https://bestprops.com`) |
| `WP_USER` | WordPress username |
| `WP_APP_PASSWORD` | WordPress application password |

### Optional - Pre-configured Logos

Any env var starting with `LOGO_` will be available as a logo key:

| Variable | Usage |
|----------|-------|
| `LOGO_TRADEIFY` | `https://your-cdn.com/tradeify-logo.png` |
| `LOGO_BESTPROPS` | `https://your-cdn.com/bestprops-logo.png` |
| `LOGO_TRADEIFY_CRYPTO` | `https://your-cdn.com/tradeify-crypto-logo.png` |

Then reference in tool calls as `logo: "tradeify"`, `logo: "bestprops"`, etc.

**Important**: Logo images should be PNG with transparent backgrounds for best results.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Server port |

## Deployment on Railway

1. Create a new project on Railway
2. Connect your GitHub repo (or push this code to a new repo)
3. Set environment variables in Railway dashboard:
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`
   - `LOGO_TRADEIFY=https://cdn.prod.website-files.com/...your-logo.png`
   - (Optional) `WP_URL`, `WP_USER`, `WP_APP_PASSWORD`
4. Railway will auto-detect the Dockerfile and deploy
5. Note your deployment URL (e.g. `https://image-gen-mcp-server-production.up.railway.app`)

## Connecting to Claude.ai

1. Go to **Claude.ai → Settings → Integrations**
2. Add a new MCP integration
3. Enter the MCP server URL: `https://your-deployment.up.railway.app/mcp`
4. Name it something like "Image Generator"

## Example Usage in Claude

Once connected, you can ask Claude things like:

- "Generate a blog header for my article about mobile DOM alerts for futures trading"
- "Create an image with the Tradeify logo for the new blog post about trailing drawdown"
- "Generate a header image and upload it to WordPress for this BestProps article"

Claude will handle prompt optimization and call the appropriate tool.

## Local Development

```bash
npm install
npm run dev
```

## Architecture

```
image-gen-mcp-server/
├── src/
│   └── index.ts          # Main server: tools, providers, compositing, transport
├── package.json
├── tsconfig.json
├── Dockerfile
└── README.md
```

## Health Check

`GET /health` returns provider status:

```json
{
  "status": "ok",
  "providers": {
    "openai": true,
    "gemini": true,
    "wordpress": true
  },
  "logos": ["tradeify", "bestprops"]
}
```
