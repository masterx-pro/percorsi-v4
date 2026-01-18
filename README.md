# Percorsi Pro v3.1 - Android Route Optimizer

🗺️ App Android per ottimizzazione percorsi con gestione Excel robusta.

## ✨ Features

- 📊 Caricamento Excel (.xlsx) e CSV
- 🔄 Ottimizzazione TSP (Nearest Neighbor + 2-opt)
- 👥 Multi-operatore con divisione geografica
- 🗺️ Mappa integrata
- 📤 Export: Excel, GPX, KML, Google Maps

## 📱 Download APK

1. Vai nella sezione **Actions** di questo repository
2. Clicca sull'ultima build completata (✓ verde)
3. Scorri fino a **Artifacts**
4. Scarica `percorsi-pro-apk`

## 🛠️ Build Manuale

L'APK viene compilato automaticamente ad ogni push su `main`.

Per triggare manualmente:
1. Vai in **Actions**
2. Seleziona **Build Android APK**
3. Clicca **Run workflow**

## 📁 Struttura

```
├── .github/workflows/build.yml  # GitHub Actions config
├── src/
│   ├── main.py                  # App source code
│   └── buildozer.spec           # Android build config
└── README.md
```

## 👤 Author

**Mattia Prosperi**

## 📄 License

MIT License
