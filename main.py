from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.core.window import Window
from td_new import get_crypto_data, calculate_indicators, identify_liquidity_zones, calculate_fibonacci_levels, generate_signals

Window.clearcolor = (0.07, 0.07, 0.07, 1)  # Binance dark style

class MainWidget(BoxLayout):
    def analyze_signal(self):
        self.ids.output.text = "[color=yellow]Analyzing...[/color]"
        symbol = self.ids.symbol.text.upper()
        interval = self.ids.interval.text
        try:
            df = get_crypto_data(symbol, interval)
            df = calculate_indicators(df)
            liquidity = identify_liquidity_zones(df)
            fib = calculate_fibonacci_levels(df)
            signals = generate_signals(df, fib, liquidity)
            msg = f"[b]{signals['signal']}[/b] Signal\nConfidence: {signals['probability']:.1f}%\nTP: {signals['tp']}\nSL: {signals['sl']}"
            self.ids.output.text = f"[color=00ff00]{msg}[/color]" if signals['signal'] == "BUY" else f"[color=ff4444]{msg}[/color]"
        except Exception as e:
            self.ids.output.text = f"[color=ff0000]Error: {str(e)}[/color]"

class CryptoSignalApp(App):
    def build(self):
        return MainWidget()

if __name__ == "__main__":
    CryptoSignalApp().run()