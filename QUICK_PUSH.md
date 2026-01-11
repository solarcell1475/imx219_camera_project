# Quick Push to GitHub

Since your SSH key is already on GitHub but not configured locally, we'll use HTTPS.

## Quick Steps

### 1. Create Personal Access Token (if you don't have one)

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. **Name:** "Jetson Upload" or "Jetson Push"
4. **Expiration:** 90 days (or your preference)
5. **Scopes:** Check ✅ `repo` (this gives full repository access)
6. Click "Generate token"
7. **COPY THE TOKEN** (starts with `ghp_...`) - you won't see it again!

### 2. Push Using HTTPS

I've already switched the remote to HTTPS. Run:

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project
git push -u origin main
```

When prompted:
- **Username:** `solarcell1475`
- **Password:** `<paste your Personal Access Token here>`

The push should succeed!

## Repository

After push, your code will be at:
**https://github.com/solarcell1475/imx219_camera_project**
