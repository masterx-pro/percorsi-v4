[app]
title = Percorsi Pro
package.name = percorsipro
package.domain = org.mattiaprosperi
version = 3.1.0

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

# Requirements essenziali
requirements = python3,kivy==2.2.1,pillow,requests,openpyxl,plyer,android

# Permissions
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE

# Android config
android.api = 33
android.minapi = 24
android.ndk = 25b
android.accept_sdk_license = True

# Architetture
android.archs = arm64-v8a,armeabi-v7a

orientation = portrait
fullscreen = 0

# AndroidX
android.enable_androidx = True

[buildozer]
log_level = 2
warn_on_root = 0
