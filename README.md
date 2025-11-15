--- 
1. [異常偵測方法，結合雙向LSTM預測與重構(Spike Anomaly和Context Anomaly捕捉)，並採取masking、雙向分數做anomaly scores，並在percentage drop超過0.13則停止警報]**AER: Auto-Encoder with Regression for Time Series Anomaly Detection**
2. [Real Time Stream Data異常偵測流程，以1-batch LSTM訓練，採取雙Detector模型(寬鬆 v.s. 嚴謹)做快速retrain和推論(異常分數採aaer)，用mu+3*sigma作為閥值，並決定是否要進第二階段Detection]**ReRe: A Lightweight Real-time Ready-to-Go Anomaly Detection Approach for Time Series**
4. [AutoEncoder於異常偵測失效推論]**AUTOENCODERS FOR ANOMALY DETECTION ARE UNRELIABLE**
5. [以multi-scale transformer + weighted attention(以feature做sigmoid對時間做weighting)的二元分類問題，模仿專家的宏觀微觀檢查序列與注意異常地方的經驗]**A Deep Learning Approach to Anomaly Detection in High-Frequency Trading Data**
6. [AutoEncoder或PCA對多資產的COV降維求Loading Factor，再從Loading Factor估計真正COV以達降噪與robust估計]**Machine Learning and Factor-Based Portfolio Optimization**
7. [Encoder過強，學會主要輪廓使正常異常的重構都好。Encoder採取弱化學超低頻特徵，Decoder再以每個像素的內插做細節調整，轉換思維加強約束能力]**What do we learn? Debunking the Myth of Unsupervised Outlier Detection**
8. [閥值動態調整，以循環次數和異常比例重新調整閥值，當異常比例超過時啟動重新閥值定義]**Adaptive Thresholding Heuristic for KPI Anomaly Detection**
9. [以Feature過Dense + Softmax去決定MOE比率，專家有三組：以Transformer去捕捉跨特徵的Attention、以LSTM去捕捉Fraudulent的持續動態變化行為、以AutoEncoder去記住正常交易行為]**Detecting Financial Fraud with Hybrid Deep Learning: A Mix-of-Experts Approach to Sequential and Anomalous Patterns**
