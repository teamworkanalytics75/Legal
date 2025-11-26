#!/bin/bash
cd /home/serteamwork/projects/LegalTech-MotionToSeal

echo "Checking what files are committed..."
git ls-files

echo ""
echo "Removing scripts/ directory from git..."
git rm -r --cached scripts/ 2>/dev/null || echo "No scripts directory to remove"

echo ""
echo "Staging updated .gitignore..."
git add .gitignore

echo ""
echo "Amending commit to remove sensitive files..."
git commit --amend -m "feat: Motion to Seal Pipeline - Core legal tech

Essential motion generation system:
- create_motion_local.py - Main script
- Documentation and config files
- Excludes scripts with API keys"

echo ""
echo "Pushing to GitHub..."
git push -u origin main --force

echo ""
echo "✅ Done! Check: https://github.com/teamworkanalytics75/Legal"

