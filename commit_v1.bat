@echo off
echo ========================================
echo Committing V1 Notebook for Colab
echo ========================================
echo.

echo Adding files...
git add data/ravdess_dataset.py
git add train_dry_watermelon_v1.ipynb
git add HUONG_DAN_COLAB_V1.md
git add COLAB_V1_QUICK_START.md
git add SUMMARY_V1_NOTEBOOK.md

echo.
echo Files staged:
git status --short

echo.
echo Committing...
git commit -F GIT_COMMIT_V1.txt

echo.
echo ========================================
echo Ready to push!
echo ========================================
echo.
echo Run: git push origin main
echo.
pause
