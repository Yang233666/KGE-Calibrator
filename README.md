# KGE-Calibrator
This is the code of KGE calibrator. 


\begin{table*}[t]
\caption{Training time in seconds taken to calibrate entity link prediction using different methods. Best and second-ranked results are in bold and underlined, respectively. For fair comparison, these results are obtained using CPU only. }
\label{training_time}
\centering
\resizebox{1\textwidth}{!}{%
\begin{tabular}{c|cccc|cccc|cccc|cccc|c}
\hline
\multirow{2}{*}{\textit{\textbf{Method}}} & \multicolumn{4}{c|}{\textit{\textbf{TransE}}} & \multicolumn{4}{c|}{\textit{\textbf{ComplEx}}}  & \multicolumn{4}{c|}{\textit{\textbf{DistMult}}}   & \multicolumn{4}{c|}{\textit{\textbf{RotatE}}} & \multirow{2}{*}{\textit{\textbf{Average}}} \\ \cline{2-17}
& \textit{WN18} & \textit{WN18RR} & \textit{FB15K} & \textit{FB15K-237} & \textit{WN18} & \textit{WN18RR} & \textit{FB15K} & \textit{FB15K-237} & \textit{WN18} & \textit{WN18RR} & \textit{FB15K} & \textit{FB15K-237} & \textit{WN18} & \textit{WN18RR} & \textit{FB15K} & \textit{FB15K-237} \\
\hline
\hline
PS & 50551.471 & 32130.612 & 66566.552 & 22756.968 & 44484.280 & 27740.023 & 66631.859 & 20060.975 & 48902.412 & 31739.057 & 58074.230 & 21682.032 & 46162.422 & 30198.810 & 65506.688 & 20522.725 & 40856.945 \\ 
VS & 2.857 & 1.893 & 25.357 & 3.493 & 2.661 & 1.620 & 16.228 & 3.218 & 4.114 & 1.914 & 20.779 & 3.456 & 2.656 & 1.706 & 25.995 & 3.277 & \underline{7.577} \\ 
TS & 5.235 & 3.207 & 20.037 & 6.475 & 5.063 & 3.121 & 18.825 & 6.276 & 5.180 & 3.204 & 19.734 & 6.412 & 5.456 & 3.171 & 20.646 & 6.345 & 8.649 \\ 
PTS & 3452.440 & 2123.849 & 16769.166 & 5856.000 & 3432.436 & 2122.273 & 16510.019 & 5764.345 & 3450.331 & 2120.555 & 16898.528 & 5868.468 & 3425.148 & 2113.001 & 16802.984 & 5853.287 & 7035.177 \\ \hline
KGEC & 2.269 & 1.334 & 8.778 & 3.049 & 2.295 & 1.391 & 8.738 & 2.950 & 2.267 & 1.395 & 8.914 & 3.013 & 2.911 & 1.331 & 5716.508 & 2.996 & 360.634 \\ 
KGEC+ & 2.389 & 1.429 & 9.373 & 3.250 & 2.350 & 1.431 & 9.348 & 3.205 & 2.371 & 1.488 & 9.349 & 3.231 & 2.367 & 1.426 & 9.423 & 3.207 & \textbf{4.102} \\ \hline
\end{tabular}%
}
\end{table*}