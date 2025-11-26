#!/bin/bash
cd /home/serteamwork/projects/LegalTech-MotionToSeal || exit 1

echo "Current directory: $(pwd)"
echo ""
echo "Git status:"
git status --short
echo ""
echo "Remote:"
git remote -v
echo ""
echo "Attempting push with timeout..."
timeout 30 git push -u origin main 2>&1 || {
    echo ""
    echo "❌ Push timed out or failed"
    echo ""
    echo "Alternative: Upload files manually to GitHub"
    echo "1. Go to: https://github.com/teamworkanalytics75/Legal"
    echo "2. Click 'Add file' → 'Upload files'"
    echo "3. Upload these files from: $(pwd)"
}

