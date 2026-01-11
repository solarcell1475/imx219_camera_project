# GitHub Repository Setup

## SSH Key Setup

The provided SSH key (GPR_SHA256) needs to be configured for GitHub authentication.

### Step 1: Add SSH Key to GitHub

1. Copy the SSH public key content
2. Go to GitHub → Settings → SSH and GPG keys
3. Click "New SSH key"
4. Paste the key and save

### Step 2: Configure Git SSH

The SSH key fingerprint is:
```
SHA256:NVZrbvRHZGe9rObWlKf2j7w4Y1V5C3m1+BzXrGcZW54
```

### Step 3: Create Repository on GitHub

1. Go to GitHub and create a new repository
2. Name it: `IMX219_Camera_Project` or `yolo-ai-vision-system`
3. **Do NOT** initialize with README, .gitignore, or license (we already have these)
4. Copy the repository SSH URL (e.g., `git@github.com:username/repo-name.git`)

### Step 4: Add Remote and Push

Once the repository is created, run:

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project

# Add remote repository (replace with your actual repo URL)
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Using HTTPS with Personal Access Token

If SSH doesn't work, you can use HTTPS with a Personal Access Token:

```bash
# Add remote with HTTPS
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push (will prompt for username and token)
git push -u origin main
```

## Repository Structure

The repository includes:
- Complete source code
- Configuration files
- Documentation (README.md, DEVELOPMENT_STATUS.md)
- Scripts for model conversion
- Git ignore for models and logs

## Important Notes

- Model files (`.pt`, `.onnx`, `.engine`) are in `.gitignore` (too large)
- Logs are excluded from repository
- Users need to download models on first run (automatic)
- See README.md for setup instructions

## Commit History

- **Version 2.1** (Current): YOLO11 upgrade, comprehensive documentation
- **Version 2.0**: Monitoring & statistics features
- **Version 1.0**: Initial production release
