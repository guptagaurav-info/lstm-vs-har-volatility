# How this project works

Colour key: **blue** = external data · **lavender** = code step · **red** = check, gate or decision · **green** = good outcome · **yellow** = caution · **grey** = output

## 1. Pipeline (what `make all` runs)

```mermaid
flowchart TD
    A["SOURCE: SPY daily open/high/low/close 1996–2026<br/>Yahoo via defeatbeta (ODC-BY)"]
    B["SOURCE: CBOE VIX daily close<br/>github.com/datasets/finance-vix (ODC-PDDL)"]
    I{"ingest.py<br/>sha256 matches config?"}
    A --> I
    B --> I
    I -- "no" --> X["Stop: source changed"]
    I -- "yes" --> D
    subgraph D["Build data · data.py"]
      D1["Daily variance = overnight² + Garman-Klass range"]
      D2["Target = log average variance over the NEXT 22 days"]
      D3{"Tests: features use only past days;<br/>target uses only future days"}
    end
    subgraph W["Walk-forward · pipeline.py (each January 2008–2026)"]
      W1["Train on all prior days minus a 22-day embargo"]
      W2["Fit: last month · VIX only · HAR · HAR+VIX · gradient boosting"]
      W3["Fit in PyTorch: LSTM and TCN on 60-day sequences, 3 seeds each"]
      W4["Forecast every day of the test year"]
    end
    E["Score: QLIKE, MSE, Diebold-Mariano vs HAR+VIX,<br/>VaR coverage, 15% vol-target backtest"]
    O["reports/metrics.json · predictions.csv · figures/*.png"]
    D --> W --> E --> O
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef step fill:#ece8ff,stroke:#6b5fbf,color:#1d1640
    classDef gate fill:#ffe0e0,stroke:#c03030,color:#400000
    classDef out fill:#eeeeee,stroke:#777,color:#222
    class A,B src
    class D1,D2,W1,W2,W3,W4,E step
    class I,X,D3 gate
    class O out
```

![Pipeline](../reports/figures/flow-1.png)

## 2. How a risk committee would choose the forecaster

```mermaid
flowchart TD
    Q["Use: position sizing, VaR, margin or option pricing?"]
    C{"Does the challenger beat HAR + VIX out of sample<br/>(Diebold-Mariano, every regime)?"}
    K["Keep HAR + VIX in production:<br/>transparent, stable, cheap to validate"]
    N["Run the neural model in shadow;<br/>re-test each year"]
    P["Promote only with a significant, stable gain<br/>and a model-risk sign-off"]
    S["Add a VIX floor and a jump overlay for crises:<br/>every model is late in the first week"]
    Q --> C
    C -- "no" --> K --> S
    C -- "yes, but unstable" --> N --> S
    C -- "yes, stable" --> P --> S
    classDef step fill:#ece8ff,stroke:#6b5fbf,color:#1d1640
    classDef gate fill:#ffe0e0,stroke:#c03030,color:#400000
    classDef good fill:#dff5e3,stroke:#2e8b57,color:#0d3318
    classDef warn fill:#fff4d6,stroke:#c9a227,color:#3d2f00
    class Q step
    class C gate
    class K,P good
    class N,S warn
```

![Decision flow](../reports/figures/flow-2.png)
