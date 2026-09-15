# Transient inlet-velocity optimization

リポジトリのルートから実行します。

```bash
uv run python -m examples.optimize.inlet_velocity_transient.run
# 更新回数・物理時間ステップ数・学習率を変更する場合
uv run python -m examples.optimize.inlet_velocity_transient.run --n-opt 6 --n-time 4 --lr 0.2
```

コードは `run.py` の `main()` から読むと、場の準備・最適化・出力の順に追えます。
定常の入口速度最適化と同じ流路・目的関数で、`PisoStepMap` を短い軌跡として
展開します。

| ファイル | 役割 |
| --- | --- |
| `config.yaml` | メッシュ、境界条件、PISO・線形ソルバーの設定 |
| `run.py` | 物理時間ステップの展開、目的関数、SGD の更新ループ |
| `output.py` | 評価レコード、CSV 保存、履歴・スライス描画 |

`config.yaml` の滑り壁流路で、固定の `deltaT` を数回進めたあとの全セル
x 方向速度の **算術平均** を 1.2 m/s に近づけます。目的関数は
`loss = (mean_u_x - 1.2)**2`、設計変数は一様な入口速度です。
平均は体積による重み付けをしません。

これは実行した有限ステップ列の autograd graph を残す微分です。
`state.checkpoint()` を学習ステップの間に挟む checkpoint–recompute 随伴では
ありません。ステップ数を増やすとメモリと計算時間が線形に増えます。
`PimpleStepMap` も同じ呼び出し形ですが、ケースは `residualControl: {}`
である必要があります。

同じ optimize example の `inlet_velocity_steady` に合わせ、設定は
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
`mapping.step()` の返り値に最終流れ場が入ります。ステップ写像は呼び出し元の
グリッドを元に戻すので、出力前に返り値の U・p を明示的にグリッドへ設定します。

既定の `deltaT = 0.5` と `--n-time 4` で、流路の対流時間（長さ 2 m /
速度 ~1 m/s）と同程度の物理時間になります。入口の影響が全域の平均に
届くようにするためです。`endTime` は最適化ループを制御しません。

スライスは既存例と同じ PyVista の平行投影・セル境界・`coolwarm` を使います。
この滑り壁流路の解はほぼ一様な速度とゼロ圧力なので、数値誤差を強調しないよう
色範囲を Ux は 0〜1.3 m/s、p は −0.01〜0.01 m²/s² に固定しています。
`p` は密度で割った圧力です。範囲は `output.py` の `U_FIELD_CLIM` と
`P_FIELD_CLIM` で変更できます。VTU には元の値を保存します。
