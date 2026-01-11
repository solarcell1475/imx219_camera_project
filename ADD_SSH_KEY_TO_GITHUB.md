# Add SSH Key to GitHub

## Your SSH Public Key

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOms8Rv2ItP7caURX+1hxW+z6azgJuG4XwSh/E1qpTSM kubong@gmail.com
```

## Steps to Add to GitHub

1. **Copy the SSH public key above** (the entire line starting with `ssh-ed25519`)

2. **Go to GitHub Settings:**
   - Visit: https://github.com/settings/keys
   - Or: GitHub → Your Profile → Settings → SSH and GPG keys

3. **Add New SSH Key:**
   - Click **"New SSH key"** button
   - **Title:** Enter a descriptive name (e.g., "Jetson Orin Nano Super")
   - **Key type:** Select "Authentication Key"
   - **Key:** Paste the entire SSH public key above
   - Click **"Add SSH key"**

4. **Verify:**
   - The key should appear in your list
   - Fingerprint should match: `SHA256:NVZrbvRHZGe9rObWlKf2j7w4Y1V5C3m1+BzXrGcZW54`

5. **Test Connection:**
   ```bash
   ssh -T git@github.com
   ```
   You should see: `Hi USERNAME! You've successfully authenticated...`

## After Adding the Key

Once the key is added to GitHub, you can:

1. **Create a new repository on GitHub:**
   - Go to: https://github.com/new
   - Repository name: `IMX219_Camera_Project` (or your preferred name)
   - Description: "YOLO AI Vision System for Jetson Orin Nano Super"
   - Keep it **Public** or **Private** (your choice)
   - **DO NOT** check "Initialize with README" (we already have files)
   - Click "Create repository"

2. **Push your code:**
   ```bash
   cd /home/jetson/Downloads/IMX219_Camera_Project
   
   # Add remote (replace USERNAME and REPO_NAME with your actual values)
   git remote add origin git@github.com:USERNAME/REPO_NAME.git
   
   # Or if remote already exists, update it:
   git remote set-url origin git@github.com:USERNAME/REPO_NAME.git
   
   # Push to GitHub
   git push -u origin main
   ```

## Troubleshooting

If SSH connection fails:

1. **Check SSH agent:**
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519  # or id_rsa if that's your key
   ```

2. **Test connection:**
   ```bash
   ssh -T git@github.com
   ```

3. **Verify key is added to GitHub:**
   - Go to: https://github.com/settings/keys
   - Confirm your key is listed

4. **Check SSH config:**
   ```bash
   cat ~/.ssh/config
   ```
   (Should not be necessary for default GitHub setup)
