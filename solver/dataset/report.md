# Ladder dataset — raport weryfikacyjny

Wygenerowano: 2026-08-23T17:13:49+00:00  
Wartości: surowe kbps per sekunda (bez wygładzania).

## Drabinki (poziom -> punkt dekodowania)

- **L3T3_tf**: l1=S0T0 | l2=S0T1 | l3=S1T1 | l4=S1T2 | l5=S2T2
- **L3T3_sf**: l1=S0T0 | l2=S1T0 | l3=S1T1 | l4=S2T1 | l5=S2T2
- **L2T2_tf**: l1=S0T0 | l2=S0T1 | l3=S1T1
- **L2T2_sf**: l1=S0T0 | l2=S1T0 | l3=S1T1

## Statystyki per poziom [kbps]

### bbb_L3T3_tf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 70.2 | 70.8 | 94.5 |
| l2 | 93.7 | 95.1 | 138.2 |
| l3 | 364.6 | 374.1 | 580.6 |
| l4 | 523.6 | 537.4 | 790.0 |
| l5 | 1920.1 | 2001.3 | 2655.3 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### bbb_L3T3_sf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 70.2 | 70.8 | 94.5 |
| l2 | 273.2 | 281.5 | 374.0 |
| l3 | 364.6 | 374.1 | 580.6 |
| l4 | 1332.2 | 1398.3 | 1876.7 |
| l5 | 1920.1 | 2001.3 | 2655.3 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### bbb_L2T2_tf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 269.3 | 265.9 | 330.5 |
| l2 | 421.5 | 413.5 | 530.9 |
| l3 | 1931.5 | 1908.3 | 2254.8 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### bbb_L2T2_sf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 269.3 | 265.9 | 330.5 |
| l2 | 1240.8 | 1235.8 | 1386.9 |
| l3 | 1931.5 | 1908.3 | 2254.8 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### johnny_L3T3_tf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 75.1 | 75.9 | 90.0 |
| l2 | 97.8 | 97.5 | 118.5 |
| l3 | 388.2 | 392.4 | 442.8 |
| l4 | 554.0 | 558.6 | 615.6 |
| l5 | 2062.6 | 2079.6 | 2232.9 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### johnny_L3T3_sf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 75.1 | 75.9 | 90.0 |
| l2 | 298.5 | 306.0 | 339.6 |
| l3 | 388.2 | 392.4 | 442.8 |
| l4 | 1448.3 | 1476.3 | 1583.7 |
| l5 | 2062.6 | 2079.6 | 2232.9 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### johnny_L2T2_tf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 266.8 | 267.1 | 283.9 |
| l2 | 414.2 | 413.2 | 438.1 |
| l3 | 1927.8 | 1932.1 | 1985.2 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

### johnny_L2T2_sf

| poziom | mean | median | p95 |
|---|---|---|---|
| l1 | 266.8 | 267.1 | 283.9 |
| l2 | 1243.9 | 1248.4 | 1279.5 |
| l3 | 1927.8 | 1932.1 | 1985.2 |

Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **0**

## Nagrania

- **bbb_L3T3**: 15427 ramek, 180.0 s, maxBitrate None kbps, pętle 1, API RTCRtpScriptTransform, karta w tle: False; klatki kluczowe w sekundach [0, 72, 87, 147, 157], skok vs mediana sekundy: 0.4454021426287176
- **bbb_L2T2**: 10716 ramek, 180.0 s, maxBitrate None kbps, pętle 1, API RTCRtpScriptTransform, karta w tle: False; klatki kluczowe w sekundach [0, 100], skok vs mediana sekundy: 0.7037151936411581
- **johnny_L3T3**: 5395 ramek, 60.0 s, maxBitrate None kbps, pętle 1, API RTCRtpScriptTransform, karta w tle: False; klatki kluczowe w sekundach [0], skok vs mediana sekundy: 0.6686183503143917
- **johnny_L2T2**: 3598 ramek, 60.1 s, maxBitrate None kbps, pętle 1, API RTCRtpScriptTransform, karta w tle: False; klatki kluczowe w sekundach [0], skok vs mediana sekundy: 0.7046932484218851

## Wykresy

Po jednym PNG na źródło x wariant w `plots/` (surowe r_raw, linia na poziom).

Uwaga: pierwsze sekundy każdego nagrania zawierają rozbieg BWE (bitrate rośnie do targetu) oraz skok klatki kluczowej — to celowo zachowane, surowe zachowanie enkodera.