[app]
title = Crypto Signal Bot
package.name = cryptosignal
package.domain = org.yourname.crypto
source.include_exts = py,png,kv,ttf,txt,json
source.dir = .
version = 0.1
requirements = kivy,requests,numpy,pandas,scipy,sklearn,textblob,nltk
orientation = portrait
fullscreen = 1
android.permissions = INTERNET

[buildozer]
log_level = 2
warn_on_root = 1