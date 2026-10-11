# Lumina Studio

A phone-first photo and cinema colour-grading web app, plus a "Fun Zone" of photo games for everyone.
Plain HTML, CSS and JavaScript (WebGL). No build step, no dependencies.

## Run it
- Open the folder in VS Code and use the Live Server extension, or run `python -m http.server 8000` and open http://localhost:8000
- Needs a browser with WebGL2 (recent Chrome, Edge, Firefox, Safari).

## Put it on your phone
1. Push this repo to GitHub, then Settings > Pages > Deploy from branch `main` / root.
2. Open the Pages link in Chrome on your phone, then menu > Install app (it works offline after the first load).

## Android app (APK)
Open the `android/` folder in Android Studio, let Gradle sync, then Build > Build APK(s).
The web files live in `android/app/src/main/assets/www`. Copy updated `index.html`, `main.js` and `styles.css` there before rebuilding.

## Notes and caveats
- Film looks, famous palettes and the film-lab simulation are approximations, not licensed stock data. Film names are used only as references.
- Camera specs and prices in the Fun Zone are US launch figures from memory, converted at rough rates. They can be wrong or out of date.
- The AI tools (subject mask, depth relight) download models from the internet on first use.
- Tickets, looks and edits are stored in the browser on the device only.
- Much of this code was written quickly and has not been fully tested. Expect bugs and please report them.

## Privacy
- No accounts and no server. Photos never leave the device.
- Stored locally in the browser: last photo and edit (IndexedDB), tickets, saved looks and flags (localStorage), app files (service worker cache).
- Network use: Google Fonts; AI model downloads (jsDelivr, Google storage, Hugging Face) only when those tools are used; shop-price links only when tapped.
- Export > Privacy & storage shows usage and clears data.
