[app]
title = Percorsi Pro
package.name = percorsipro
package.domain = org.mattiaprosperi
version = 3.2.0

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

requirements = python3,kivy==2.2.1,pillow,requests,openpyxl,et_xmlfile,plyer,android

android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE

android.api = 33
android.minapi = 24
android.ndk = 25b
android.accept_sdk_license = True

android.archs = arm64-v8a,armeabi-v7a

orientation = portrait
fullscreen = 0

android.enable_androidx = True

p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 0
