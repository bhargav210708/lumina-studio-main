package com.lumina.studio
import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.webkit.*
import android.widget.Toast
import androidx.webkit.WebViewAssetLoader

class MainActivity : Activity() {
  private lateinit var web: WebView
  private var chooser: ValueCallback<Array<Uri>>? = null
  inner class Bridge {
    @JavascriptInterface fun save(b64: String, name: String, mime: String) {
      val v = ContentValues().apply { put(MediaStore.Downloads.DISPLAY_NAME, name); put(MediaStore.Downloads.MIME_TYPE, mime) }
      val u = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v) ?: return
      contentResolver.openOutputStream(u)?.use { it.write(Base64.decode(b64, Base64.DEFAULT)) }
      runOnUiThread { Toast.makeText(this@MainActivity, "Saved to Downloads: $name", Toast.LENGTH_LONG).show() }
    }
  }
  override fun onCreate(s: Bundle?) {
    super.onCreate(s)
    requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA), 2)
    val loader = WebViewAssetLoader.Builder().addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this)).build()
    web = WebView(this).apply {
      setBackgroundColor(Color.BLACK)
      settings.javaScriptEnabled = true; settings.domStorageEnabled = true; settings.mediaPlaybackRequiresUserGesture = false
      addJavascriptInterface(Bridge(), "LuminaNative")
      webViewClient = object : WebViewClient() {
        override fun shouldInterceptRequest(v: WebView, r: WebResourceRequest) = loader.shouldInterceptRequest(r.url)
      }
      webChromeClient = object : WebChromeClient() {
        override fun onPermissionRequest(r: PermissionRequest) { r.grant(r.resources) }
        override fun onShowFileChooser(w: WebView, cb: ValueCallback<Array<Uri>>, p: FileChooserParams): Boolean {
          chooser?.onReceiveValue(null); chooser = cb; startActivityForResult(p.createIntent(), 1); return true
        }
      }
      loadUrl("https://appassets.androidplatform.net/assets/www/index.html")
    }
    setContentView(web)
  }
  override fun onActivityResult(rc: Int, res: Int, d: Intent?) {
    if (rc == 1) { chooser?.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(res, d)); chooser = null }
  }
  override fun onBackPressed() { if (web.canGoBack()) web.goBack() else super.onBackPressed() }
}
