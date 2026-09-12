## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-22
- Verification Status: VERIFIED
- Version Label: shi_player_profile_validation_v2.0

## Validation Report

- **Source**: shi-player-profile-v2
- **Overall Confidence**: CAUTION
- **Reproducibility Verdict**: REPRODUCIBLE

`CAUTION` 表示结果可复现、方法边界清楚，但样本只有24个比赛配对，多指标估计精度有限，且部分指标对加权方式或单场删除敏感；不表示程序运行失败。

### Statistical Findings

| Metric | Estimate | 95% Bootstrap CI | Tendency score | Stability | Confidence |
|---|---:|---:|---:|---|---|
| 前三拍主动类别比例 | +0.0757 | [0.0179, 0.1381] | 66.7 | STABLE | CAUTION |
| 空间使用多样性 | +0.0189 | [-0.0018, 0.0425] | 62.5 | STABLE | CAUTION |
| 吊球倾向 | +0.0039 | [-0.0053, 0.0125] | 62.5 | STABLE | CAUTION |
| 抽挡倾向 | +0.0081 | [-0.0178, 0.0335] | 58.3 | STABLE | CAUTION |
| 终局后场进攻事件比例 | -0.0070 | [-0.0267, 0.0120] | 45.8 | STABLE | CAUTION |
| 其余八项指标 | 区间均跨0 | — | — | 部分UNSTABLE | CAUTION |

解释：当前13项配对指标中，只有“前三拍主动类别比例”的未校正Bootstrap区间不含0。该差值约为7.57个百分点，石宇奇在16/24场高于同场对手。由于同时查看了13项指标，且其中9场石宇奇记录低于该指标的每场50拍警戒阈值，该结果应写成探索性描述，不应写成确认性优势。

### Numerical and Design Checks

- 13项配对指标均有24个有效比赛配对。
- 所有比例、熵和倾向分数均在允许范围内。
- 所有Bootstrap区间上下限顺序正确。
- 未运行或筛选p值；主要推断基于原始差值和比赛级Bootstrap区间。
- Bootstrap以比赛为单位并保持同场配对，没有把逐拍事件作为独立样本。
- 9场 `early_active_share` 和4场 `defense_to_attack_proxy` 被标记为 `LOW_N`；这些比赛仍保留，避免选择性排除。
- 4,076/14,157（28.8%）石宇奇事件为unknown；有效预测落点6,459/14,157（45.6%）。S2与S7已覆盖相应口径敏感性。

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Multiplicity | 同时观察13项配对区间，没有多重检验校正 | 尤其是前三拍主动类别比例 |
| Weighting reversal | 比赛等权与事件合并口径在网前、后场、序列、进攻进入和网前连续使用上方向相反 | 5项指标 |
| Single-match influence | 删除任一比赛时，6项微小配对差至少一次改变方向 | 相关UNSTABLE指标 |
| Construct boundary | 过程代理没有回合胜负、失误或得分结果 | 五项过程代理 |
| Sampling boundary | 24场均含石宇奇，不构成随机抽取的全部职业比赛 | 所有外推结论 |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Finding | Control in current analysis |
|---|---|---|---|
| 1. Simpson's paradox | RED_FLAG for pooled interpretation | 5项指标在比赛等权与事件合并后方向反转 | 主分析固定比赛等权；反转指标标记UNSTABLE |
| 2. Ecological fallacy | CAUTION | 比赛级结果不能推出每一拍或石宇奇全部生涯行为 | 推断总体限定为当前24场 |
| 3. Berkson's paradox | CAUTION | 样本只含已进入数据集且有石宇奇的比赛，存在选择机制 | 不把样本称为随机或世界常模 |
| 4. Collider bias | NOTE | 未建立含控制变量的回归模型，当前无可识别的碰撞变量调整 | 不适用；未来加入比分/赛事控制时重查 |
| 5. Base-rate neglect | NOTE | 未报告诊断准确率、PPV或NPV | 不适用 |
| 6. Regression to the mean | NOTE | 非极端值筛选的前后测设计 | 不适用 |
| 7. Survivorship bias | CAUTION | 无法证明24场覆盖所有符合条件比赛；未纳入比赛的机制未知 | 结论限定于纳入比赛 |
| 8. Look-elsewhere effect | CAUTION | 同时检查13项指标，但没有只报告区间不含0的结果 | 全部指标、区间和敏感性结果均公开；不作确认性显著结论 |
| 9. Garden of forking paths | NOTE | v1已查看全部数据，无法事前注册 | v1标为pilot；v2规则锁定并报告S1–S9 |
| 10. Correlation ≠ causation | CAUTION if causalized | 观察性描述不能说明某打法导致得分或胜负 | 报告禁止因果和比赛效果措辞 |
| 11. Reverse causality | NOTE | 当前没有方向性因果模型 | 不适用；不使用“导致/改善”语言 |

### Reproducibility

- **Method**: deterministic full re-run with the same command, data, script, protocol, environment, seed 20260822 and 10,000 Bootstrap repetitions.
- **Runs**: 2/2 completed with exit code 0.
- **Compared artifacts**: 18.
- **Comparison**: file count, byte size and SHA-256.
- **Mismatches**: 0.
- **Verdict**: REPRODUCIBLE.

### Validated Interpretation Boundary

可支持的核心表述是：在当前24场算法标注比赛中，石宇奇前三拍主动类别比例的比赛等权均值为0.629，同场对手为0.554；平均配对差为0.0757，Bootstrap 95%区间为[0.0179, 0.1381]。这描述的是前三拍球种结构，不是开局得分能力或接发质量。

除该项外，其余配对区间均跨0；其中部分指标还对加权或单场删除敏感。因此，当前论文更适合强调“画像框架、比赛间变异和方法透明性”，而不是宣称石宇奇在多个维度系统性优于对手。
