# Matched and crossed saved peak ratios

Raw relative errors in percent. U=both peaks usable; W=one or both withheld. Pass 6 is fixed. Crossed pairs are dependent recombinations of two saved realizations, not additional independent trials.

| Arm | Array | Ratio | Seed 1 / 1 | Seed 2 / 2 | Cross 1 / 2 | Cross 2 / 1 |
| --- | --- | --- | --- | --- | --- | --- |
| P | a1100 | H/D | -0.1279 (U) | -0.1687 (W) | +2.1656 (U) | -2.4098 (W) |
| P | a1100 | D/H | +0.1281 (U) | +0.1690 (W) | +2.4693 (W) | -2.1197 (U) |
| P | a1100 | T/H | +0.0000 (U) | +0.0000 (W) | +2.3382 (W) | -2.2848 (W) |
| P | a1400 | H/D | +5.9616 (W) | +1.5611 (W) | +8.1722 (W) | -0.5143 (W) |
| P | a1400 | D/H | -5.6262 (W) | -1.5371 (W) | +0.5170 (W) | -7.5548 (W) |
| P | a1400 | T/H | +0.8168 (W) | -0.0000 (W) | +7.3794 (W) | -6.1116 (W) |
| P | a2000 | H/D | +3.4669 (U) | +2.6299 (W) | +16.1827 (W) | -8.6025 (U) |
| P | a2000 | D/H | -3.3508 (U) | -2.5625 (W) | +9.4122 (U) | -13.9287 (W) |
| P | a2000 | T/H | +0.0000 (U) | +0.0000 (U) | +13.2055 (U) | -11.6650 (U) |
| C | a1100 | H/D | -0.2145 (W) | -0.7663 (U) | +2.1093 (U) | -3.0246 (W) |
| C | a1100 | D/H | +0.2150 (W) | +0.7722 (U) | +3.1190 (W) | -2.0657 (U) |
| C | a1100 | T/H | +0.0000 (U) | +0.0000 (U) | +2.8977 (U) | -2.8161 (U) |
| C | a1400 | H/D | +0.5240 (W) | -0.2010 (W) | +5.6474 (W) | -5.0409 (W) |
| C | a1400 | D/H | -0.5212 (W) | +0.2015 (W) | +5.3085 (W) | -5.3456 (W) |
| C | a1400 | T/H | -1.3211 (U) | +0.8071 (W) | +4.4618 (W) | -4.7735 (W) |
| C | a2000 | H/D | -0.5350 (W) | +0.8004 (W) | +7.9047 (W) | -7.0836 (W) |
| C | a2000 | D/H | +0.5379 (W) | -0.7941 (W) | +7.6236 (W) | -7.3256 (W) |
| C | a2000 | T/H | +0.0000 (U) | +0.0000 (U) | +7.0478 (U) | -6.5838 (U) |

## How common seed response cancels in matched H/D

Let h=H2/H1 and d=D2/D1. The ratio of matched gains is h/d; H1/D2 divided by H1/D1 is 1/d. These are exact identities of the saved peak readouts, not a fitted noise model.

| Arm | Array | H seed change % | D seed change % | Matched gain change % | H width-area change % | D width-area change % | H fitted background change | D fitted background change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P | a1100 | -2.2848 | -2.2449 | -0.0408 | +4.7685 | +5.8652 | +0.3871 | +0.3929 |
| P | a1400 | -6.1116 | -2.0436 | -4.1529 | -6.2722 | -8.8353 | +0.5765 | +0.6151 |
| P | a2000 | -11.6650 | -10.9446 | -0.8090 | +5.5627 | +6.9838 | -0.1182 | -0.0989 |
| C | a1100 | -2.8161 | -2.2758 | -0.5529 | +6.9555 | +5.9215 | +0.2263 | +0.2419 |
| C | a1400 | -5.5359 | -4.8496 | -0.7212 | -6.1117 | +5.6670 | +0.0626 | -1.0104 |
| C | a2000 | -6.5838 | -7.8214 | +1.3426 | +3.4644 | +3.2415 | -0.5590 | -0.5798 |
