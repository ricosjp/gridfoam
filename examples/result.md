# Drag comparison result

`examples/low_re_sphere/compare.py`,
`examples/low_re_cylinder/compare.py`,
`examples/low_re_cube/compare.py` の実行結果を整理した。

## Summary

- sphere は gridfoam と OpenFOAM の差が比較的小さい。Re <= 10 では gridfoam が OpenFOAM より約 3.1-7.8% 高く、Re=100 では約 10.1% 低い。
- cylinder は gridfoam が OpenFOAM より一貫して高い。差は低 Re で大きく、Re=0.5 で約 40.6%、Re=1 で約 39.0%、Re=100 で約 4.5% まで縮小する。
- cube は低 Re で gridfoam が OpenFOAM より大きく高い。Re=0.5 で約 71.3%、Re=1 で約 66.7% 高い。一方、Re=100, 500 では gridfoam の方が低い。
- sphere と cylinder は OpenFOAM も paper correlation より高い。したがって paper との差は gridfoam 固有の誤差だけではなく、計算領域、境界条件、ブロッケージ、参照面積・代表長さ、論文相関の適用条件との差を含む。
- gridfoam と OpenFOAM の差は、sphere より cylinder/cube で顕著である。形状依存性が強いため、AP-IBM の投影面積・法線・壁面応力評価、または抗力係数の正規化条件の差を優先して疑うべきである。

## Sphere

Output label は `Cheng 2009` だが、`compare.py` 内の相関関数コメントは `Clift et al. (1978)` になっている。paper 欄の出典名は先に整理する必要がある。

| Re | Cd(gridfoam) | Cd(OpenFOAM) | Cd(paper) | gridfoam/OpenFOAM | gridfoam/paper |
|---:|-------------:|-------------:|----------:|------------------:|---------------:|
| 0.5 | 92.5324 | 85.8320 | 51.4905 | 1.078 | 1.797 |
| 1 | 46.4687 | 43.1044 | 27.1560 | 1.078 | 1.711 |
| 2 | 23.4310 | 21.7790 | 14.7197 | 1.076 | 1.592 |
| 5 | 9.67325 | 9.08765 | 6.87527 | 1.064 | 1.407 |
| 10 | 5.12530 | 4.97361 | 3.99959 | 1.031 | 1.281 |
| 100 | 1.17614 | 1.30792 | 1.08702 | 0.899 | 1.082 |

## Cylinder

| Re_D | Cd(gridfoam) | Cd(OpenFOAM) | Cd(paper) | gridfoam/OpenFOAM | gridfoam/paper |
|----:|-------------:|-------------:|----------:|------------------:|---------------:|
| 0.5 | 71.2946 | 50.7167 | 16.5533 | 1.406 | 4.307 |
| 1 | 35.8214 | 25.7786 | 9.93960 | 1.390 | 3.604 |
| 2 | 18.3258 | 13.4404 | 6.23451 | 1.363 | 2.939 |
| 5 | 7.78528 | 6.24291 | 3.67794 | 1.247 | 2.117 |
| 10 | 4.47127 | 3.90820 | 2.67182 | 1.144 | 1.674 |
| 100 | 1.67724 | 1.60578 | 1.44900 | 1.045 | 1.158 |

## Cube

| Re | Cd(gridfoam) | Cd(OpenFOAM) | gridfoam/OpenFOAM |
|---:|-------------:|-------------:|------------------:|
| 0.5 | 112.095 | 65.4463 | 1.713 |
| 1 | 56.2514 | 33.7436 | 1.667 |
| 2 | 28.4025 | 18.0653 | 1.572 |
| 5 | 11.7593 | 8.67148 | 1.356 |
| 10 | 6.26669 | 5.38041 | 1.165 |
| 100 | 1.52034 | 1.80597 | 0.842 |
| 500 | 1.12431 | 1.26785 | 0.887 |

## Cause analysis

### 1. Paper correlation との差は gridfoam 単独の問題ではない

sphere と cylinder では OpenFOAM も paper correlation より高い。特に cylinder の低 Re では OpenFOAM/paper も約 2.6-3.1 倍であり、paper との差をそのまま gridfoam の離散化誤差とはみなせない。

低 Re の外部流れ抗力は遠方境界の距離、側方境界のブロッケージ、入口・出口条件、2D/3D の扱いに敏感である。Stokes 領域では擾乱が遠方まで減衰しにくいため、計算領域が有限であるだけで抗力が上がりやすい。まず OpenFOAM ケースと paper correlation の前提条件が一致しているかを確認する必要がある。

### 2. gridfoam と OpenFOAM の差は形状依存性が強い

sphere は比較的近いが、cylinder と cube では低 Re で gridfoam が大きく高い。この傾向は、単純な grad/interpolation scheme の問題だけでは説明しにくい。

疑うべき優先度が高いのは次の項目である。

- AP-IBM の投影面積が物理表面積または投影面積を過大評価している。
- AP face の法線方向、面積ベクトル、owner/neighbor 側の符号が、非球形状や sharp edge で OpenFOAM の壁面積分とずれている。
- cylinder/cube の immersed boundary 近傍で壁面せん断応力を過大評価している。
- 抗力係数の正規化に使う `A_ref`, `L_ref`, `U_ref`, `rho` が OpenFOAM 側と完全一致していない。

特に cube は低 Re で gridfoam/OpenFOAM が 1.6-1.7 程度だが、Re=100, 500 では 0.84-0.89 に反転している。これは単なる定数倍の参照面積ミスだけではなく、低 Re の粘性抗力評価と高 Re の圧力抗力・剥離評価の両方に差がある可能性を示す。

### 3. 低 Re では粘性項と壁面勾配の影響が支配的

低 Re では抗力に占める粘性寄与が大きい。octree の階層差、skewness、immersed boundary 近傍の再構成誤差が `grad(U)` や `snGrad(U)` に入ると、抗力に直接反映される。

今回の reconstruction 修正により、octree 階層差で速度分布が障害物のように見える問題は改善対象になったが、抗力値ではまだ壁面近傍の勾配評価と力積分が支配的に残っている可能性が高い。したがって、体積場の見た目だけでなく、表面積分される圧力・粘性応力を分解して比較する必要がある。

### 4. Mesh と geometry representation が同一ではない

gridfoam の octree/AP-IBM mesh と OpenFOAM の mesh は同じ形状を解いていても、壁面表現、表面解像度、境界層のセル配置、wake 解像度が異なる。sphere で差が小さく、cube/cylinder で差が大きいことから、形状表現と壁面近傍の解像度差は主要因の候補である。

## Recommended verification

1. **Reference normalization audit**

   各ケースで `rho`, `U_ref`, `L_ref`, `A_ref` と実抗力 `F_D` を出力し、gridfoam と OpenFOAM の `Cd = F_D / (0.5 rho U_ref^2 A_ref)` が同じ定義になっているか確認する。

2. **Force decomposition**

   pressure drag と viscous drag を分けて OpenFOAM と比較する。低 Re の差が viscous 側に集中するなら、壁面勾配・AP 面積・法線が主因である。pressure 側にも大きく出るなら、圧力境界条件、圧力補正、形状表現を調べる。

3. **AP surface audit**

   AP-IBM が生成した表面について、合計面積、抗力方向への投影面積、法線分布を出力する。sphere/cylinder/cube の解析値または OpenFOAM surface mesh と比較する。

4. **Domain-size study**

   低 Re の sphere/cylinder で外部境界を 2 倍以上に広げ、Cd が paper correlation に近づくか確認する。OpenFOAM も同じ条件で再実行し、paper との差が計算領域起因かを切り分ける。

5. **Mesh refinement study**

   物体近傍と wake の octree refinement を 1 段階ずつ上げ、Cd の収束傾向を見る。cube/cylinder の低 Re で gridfoam/OpenFOAM 比が縮むかを確認する。

6. **Convergence audit**

   `force_coeffs.csv` の Cd history、連続の式の残差、圧力・速度残差を確認する。低 Re steady case は収束が遅く、見かけ上安定していても Cd が残差に依存している場合がある。

## Implementation priorities

1. 抗力評価コードに pressure/viscous decomposition と正規化値の詳細出力を追加する。
2. AP-IBM surface の合計面積、投影面積、法線統計を検証できるテストまたは診断スクリプトを追加する。
3. cylinder/cube の低 Re ケースで `linear` と `leastsquare` の grad/interpolation/snGrad を切り替え、Cd と force decomposition の感度を見る。
4. OpenFOAM と同じ参照値・同じ領域サイズ・可能な限り同等の表面解像度で再比較する。
