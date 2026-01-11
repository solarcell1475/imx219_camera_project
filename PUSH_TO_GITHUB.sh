#!/bin/bash
# Script to push repository to GitHub
# Make sure you have:
# 1. Created a GitHub repository
# 2. Added your SSH key to GitHub (Settings → SSH and GPG keys)
# 3. Updated the REPO_URL below with your actual repository URL

set -e

# Configuration - UPDATE THIS WITH YOUR REPOSITORY URL
REPO_URL="git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git"

echo "========================================="
echo "GitHub Repository Push Script"
echo "========================================="
echo ""

# Check if remote exists
if git remote | grep -q "^origin$"; then
    echo "Remote 'origin' already exists."
    CURRENT_URL=$(git remote get-url origin)
    echo "Current URL: $CURRENT_URL"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote set-url origin "$REPO_URL"
        echo "Remote URL updated to: $REPO_URL"
    fi
else
    echo "Adding remote repository..."
    git remote add origin "$REPO_URL"
    echo "Remote 'origin' added: $REPO_URL"
fi

echo ""
echo "Current branch: $(git branch --show-current)"
echo ""

# Check SSH key
echo "Testing SSH connection to GitHub..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✓ SSH authentication successful!"
else
    echo "⚠ SSH authentication failed or not configured."
    echo "Please add your SSH key to GitHub:"
    echo "1. Go to: https://github.com/settings/keys"
    echo "2. Click 'New SSH key'"
    echo "3. Paste your public key (usually ~/.ssh/id_rsa.pub or ~/.ssh/id_ed25519.pub)"
    echo ""
    echo "Your SSH key fingerprint should be:"
    echo "SHA256:NVZrbvRHZGe9rObWlKf2j7w4Y1V5C3m1+BzXrGcZW54"
    exit 1
fi

echo ""
read -p "Ready to push to GitHub? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "========================================="
echo "✓ Successfully pushed to GitHub!"
echo "========================================="
echo ""
echo "Repository URL: https://github.com/$(echo $REPO_URL | sed 's/.*github.com[:/]\(.*\)\.git/\1/')"
echo ""
