#!/bin/bash
# Push IMX219 Camera Project to GitHub
# Repository: https://github.com/solarcell1475/imx219_camera_project

set -e

REPO_URL="https://github.com/solarcell1475/imx219_camera_project.git"
REPO_USER="solarcell1475"

echo "========================================="
echo "Push to GitHub - IMX219 Camera Project"
echo "========================================="
echo ""
echo "Repository: $REPO_URL"
echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo "Error: Not a git repository!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Configure remote if needed
echo "Checking remote configuration..."
if git remote get-url origin &>/dev/null; then
    CURRENT_URL=$(git remote get-url origin)
    if [[ "$CURRENT_URL" != *"$REPO_USER"* ]]; then
        echo "Updating remote URL..."
        git remote set-url origin "$REPO_URL"
    fi
    echo "✓ Remote configured: $REPO_URL"
else
    echo "Adding remote..."
    git remote add origin "$REPO_URL"
    echo "✓ Remote added: $REPO_URL"
fi

echo ""

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Ensure we're on main branch
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Switching to main branch..."
    git branch -M main
    CURRENT_BRANCH="main"
fi

echo ""

# Show commit status
echo "Latest commit:"
git log --oneline -1
echo ""

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠ Warning: You have uncommitted changes!"
    echo "Files with changes:"
    git status --short
    echo ""
    read -p "Do you want to commit these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Update project files"
        echo "✓ Changes committed"
    else
        echo "Skipping commit. Pushing existing commits..."
    fi
    echo ""
fi

# Instructions for Personal Access Token
echo "========================================="
echo "GitHub Authentication Required"
echo "========================================="
echo ""
echo "You need a Personal Access Token to push to GitHub."
echo ""
echo "If you don't have one, create it here:"
echo "  https://github.com/settings/tokens"
echo ""
echo "Steps:"
echo "  1. Click 'Generate new token (classic)'"
echo "  2. Name: 'Jetson Upload'"
echo "  3. Check 'repo' scope"
echo "  4. Generate and COPY the token (starts with ghp_)"
echo ""
echo "========================================="
echo ""

# Try to push
echo "Attempting to push to GitHub..."
echo ""
echo "You will be prompted for:"
echo "  Username: $REPO_USER"
echo "  Password: <paste your Personal Access Token here>"
echo ""
echo "Note: Use your Personal Access Token, NOT your GitHub password!"
echo ""

read -p "Ready to push? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Pushing to GitHub..."
echo ""

# Configure git to use credential helper (optional, helps store token)
git config --global credential.helper store 2>/dev/null || true

# Push with error handling
if git push -u origin main; then
    echo ""
    echo "========================================="
    echo "✓ Successfully pushed to GitHub!"
    echo "========================================="
    echo ""
    echo "Repository URL:"
    echo "  https://github.com/$REPO_USER/imx219_camera_project"
    echo ""
    echo "View your repository:"
    echo "  https://github.com/$REPO_USER/imx219_camera_project"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Push failed"
    echo "========================================="
    echo ""
    echo "Possible reasons:"
    echo "  1. Invalid Personal Access Token"
    echo "  2. Token doesn't have 'repo' scope"
    echo "  3. Network connection issue"
    echo ""
    echo "To retry:"
    echo "  1. Make sure you have a valid token"
    echo "  2. Run this script again: ./push_to_github.sh"
    echo ""
    exit 1
fi
