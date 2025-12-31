# 📦 Cài đặt ffmpeg

ffmpeg cần thiết để extract audio từ video files.

---

## Windows

### **Option 1: Chocolatey (Khuyến nghị)**

```powershell
# Install Chocolatey (nếu chưa có)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install ffmpeg
choco install ffmpeg
```

### **Option 2: Manual Download**

1. Download ffmpeg từ: https://ffmpeg.org/download.html#build-windows
2. Hoặc từ: https://www.gyan.dev/ffmpeg/builds/
3. Extract file zip
4. Thêm vào PATH:
   - Mở System Properties → Environment Variables
   - Thêm đường dẫn `ffmpeg/bin` vào PATH
   - Ví dụ: `C:\ffmpeg\bin`

### **Option 3: Scoop**

```powershell
# Install Scoop (nếu chưa có)
iwr -useb get.scoop.sh | iex

# Install ffmpeg
scoop install ffmpeg
```

### **Verify Installation**

```powershell
ffmpeg -version
```

Expected output:
```
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
...
```

---

## Linux

### **Ubuntu/Debian**

```bash
sudo apt update
sudo apt install ffmpeg
```

### **Fedora/RHEL/CentOS**

```bash
sudo dnf install ffmpeg
```

### **Arch Linux**

```bash
sudo pacman -S ffmpeg
```

### **Verify Installation**

```bash
ffmpeg -version
```

---

## macOS

### **Homebrew (Khuyến nghị)**

```bash
# Install Homebrew (nếu chưa có)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg
```

### **MacPorts**

```bash
sudo port install ffmpeg
```

### **Verify Installation**

```bash
ffmpeg -version
```

---

## Troubleshooting

### **Windows: 'ffmpeg' is not recognized**

**Problem:**
```
'ffmpeg' is not recognized as an internal or external command
```

**Solution:**
1. Check ffmpeg is in PATH
2. Restart terminal/PowerShell
3. Restart computer if needed

**Manual PATH setup:**
```powershell
# Add to PATH temporarily
$env:Path += ";C:\path\to\ffmpeg\bin"

# Or add permanently via System Properties
```

### **Linux: Package not found**

**Problem:**
```
E: Unable to locate package ffmpeg
```

**Solution:**
```bash
# Enable universe repository (Ubuntu)
sudo add-apt-repository universe
sudo apt update
sudo apt install ffmpeg
```

### **Permission Denied**

**Problem:**
```
Permission denied
```

**Solution:**
```bash
# Use sudo
sudo apt install ffmpeg
```

---

## Test ffmpeg

### **Extract audio from video**

```bash
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

### **Get video info**

```bash
ffmpeg -i input.mp4
```

### **Convert video format**

```bash
ffmpeg -i input.avi output.mp4
```

---

## Alternative: Use conda

If you have conda/anaconda:

```bash
conda install -c conda-forge ffmpeg
```

---

## Summary

**Windows:**
```powershell
choco install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Verify:**
```bash
ffmpeg -version
```

**Done! ✅**
