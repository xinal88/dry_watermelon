@echo off
echo ========================================
echo Committing Simple RAVDESS Loader
echo ========================================
echo.

echo Adding files...
git add data/simple_ravdess_dataset.py
git add build_colab_notebook.py
git add train_dry_watermelon_v1.ipynb
git add FINAL_FIX_SIMPLE_LOADER.md

echo.
echo Files staged:
git status --short

echo.
echo Committing...
git commit -m "Add simple RAVDESS loader - works with ANY folder structure"

echo.
echo ========================================
echo Ready to push!
echo ========================================
echo.
echo Run: git push origin main
echo.
pause
