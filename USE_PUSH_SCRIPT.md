# How to Use push_to_github.sh

## Quick Start

1. **Create a Personal Access Token** (if you don't have one):
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Name: "Jetson Upload"
   - Check ✅ `repo` scope
   - Generate and **COPY the token** (starts with `ghp_`)

2. **Run the script:**
   ```bash
   cd /home/jetson/Downloads/IMX219_Camera_Project
   ./push_to_github.sh
   ```

3. **When prompted:**
   - Username: `solarcell1475`
   - Password: `<paste your Personal Access Token>`
   - **Important:** Use the token, NOT your GitHub password!

## What the Script Does

- ✅ Checks git repository status
- ✅ Configures remote repository URL
- ✅ Ensures you're on the `main` branch
- ✅ Shows latest commit
- ✅ Checks for uncommitted changes (offers to commit)
- ✅ Guides you through authentication
- ✅ Pushes to GitHub
- ✅ Shows success message with repository URL

## After Successful Push

Your code will be available at:
**https://github.com/solarcell1475/imx219_camera_project**

## Troubleshooting

If push fails:
1. Make sure your Personal Access Token has `repo` scope
2. Verify the token is correct (starts with `ghp_`)
3. Check your internet connection
4. Run the script again: `./push_to_github.sh`
