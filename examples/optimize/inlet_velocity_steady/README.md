# Steady inlet-velocity optimization

リポジトリのルートから実行します。

```bash
uv run python -m examples.optimize.inlet_velocity_steady.run
# 更新回数・学習率を変更する場合
uv run python -m examples.optimize.inlet_velocity_steady.run --n-opt 10 --lr 0.2
```

コードは `run.py` の `main()` から読むと、場の準備・最適化・出力の順に追えます。
温度最適化の例と同じ構成です。

| ファイル | 役割 |
| --- | --- |
| `config.yaml` | メッシュ、境界条件、SIMPLE・線形ソルバーの設定 |
| `run.py` | 定常求解、目的関数、SGD の更新ループ |
| `output.py` | 評価レコード、CSV 保存、履歴・スライス描画 |

`config.yaml` の流路で定常 SIMPLE を解き、全セルの x 方向速度の
**算術平均**を 1.2 m/s に近づけます。目的関数は
`loss = (mean_u_x - 1.2)**2`、設計変数は一様な入口速度です。
平均は体積による重み付けをしません。

同じ optimize example の `inlet_temperature_steady` に合わせ、設定は
`config.yaml`、出力は `simulator.control.output` で指定します。
標準ではこのディレクトリの `outputs/` に以下を保存します。

| ファイル | 内容 |
| --- | --- |
| `inlet_velocity_history.csv` | `step,inlet_velocity,mean_u_x,loss` |
| `inlet_velocity_history.png` | CSV と同じ評価レコードから描いた速度・loss の履歴 |
| `inlet_velocity_final.vtu` | 最後の更新後に再計算したセル中心の U と p |
| `inlet_velocity_final_fields.png` | 中央 XY 面（z = 0.05 m）の Ux と p のスライス |

`step=0` は初期評価、`step=n` は SGD を n 回更新した後の評価です。
標準の 4 回更新では 5 行を記録し、CSV 最終行と VTU は同じ状態に対応します。
各行の入口速度・平均速度・loss はすべて同じパラメータでの評価です。
`steady_solve()` の返り値に最終流れ場が入ります。ステップ写像は呼び出し元の
グリッドを元に戻すので、出力前に返り値の U・p を明示的にグリッドへ設定します。

スライスは既存例と同じ PyVista の平行投影・セル境界・`coolwarm` を使います。
この滑り壁流路の解はほぼ一様な速度とゼロ圧力なので、数値誤差を強調しないよう
色範囲を Ux は 0〜1.3 m/s、p は −0.01〜0.01 m²/s² に固定しています。
`p` は密度で割った圧力です。範囲は `output.py` の `U_FIELD_CLIM` と
`P_FIELD_CLIM` で変更できます。VTU には元の値を保存します。
