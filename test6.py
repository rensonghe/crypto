#%%
import pandas as pd
import matplotlib.pyplot as plt
# List of file names




def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    cumulative_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown



file_paths_capital = ['/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-01-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-02-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-03-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-04-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-05-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-06-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-07-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-08-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv',
                      '/home/xianglake/yiliang/log/eosusdt_pnl_withstop_2023-09-01_vwap_2s_50_60000_0.0003_0.009p_0.005l_sec_capital.csv'
              #'D:/research_log/xrp_pnl_withstop_2023-05-01_vwap_30s_80_120000_sec.csv'
]
file_paths = ['/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-01-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-02-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-03-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-04-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-05-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-06-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-07-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv',
              '/home/xianglake/yiliang/log/dogeusdt_pnl_withstop_2023-08-01_vwap_5s_70_90000_0.0003_0.009p_0.005l_sec.csv'
              #'D:/research_log/xrp_pnl_withstop_2023-05-01_vwap_30s_80_120000_sec.csv'
]
df_dict = {}
for file_path in file_paths_capital:
    file_name = file_path.split('/')[-1].split('.')[0]
    df_dict[file_name] = pd.read_csv(file_path)

# for file_path in file_paths:
#     file_name = file_path.split('/')[-1].split('.')[0]
#     df_dict[file_name] = pd.read_csv(file_path)
#
# Access the dataframes using their file names
df_a = df_dict['eosusdt_pnl_withstop_2023-01-01_vwap_2s_50_60000_0']
df_b = df_dict['eosusdt_pnl_withstop_2023-02-01_vwap_2s_50_60000_0']
df_c = df_dict['eosusdt_pnl_withstop_2023-03-01_vwap_2s_50_60000_0']
df_d = df_dict['eosusdt_pnl_withstop_2023-04-01_vwap_2s_50_60000_0']
df_e = df_dict['eosusdt_pnl_withstop_2023-05-01_vwap_2s_50_60000_0']
df_f = df_dict['eosusdt_pnl_withstop_2023-06-01_vwap_2s_50_60000_0']
df_h = df_dict['eosusdt_pnl_withstop_2023-07-01_vwap_2s_50_60000_0']
df_g = df_dict['eosusdt_pnl_withstop_2023-08-01_vwap_2s_50_60000_0']
df_i = df_dict['eosusdt_pnl_withstop_2023-09-01_vwap_2s_50_60000_0']

# df_d = df_dict['xrp_pnl_withstop_2023-05-01_vwap_30s_80_120000_sec']

r_a=df_a.iloc[-1,6]/df_a.iloc[0,6]
r_b=df_b.iloc[-1,6]/df_b.iloc[0,6]
r_c=df_c.iloc[-1,6]/df_c.iloc[0,6]
r_d=df_d.iloc[-1,6]/df_d.iloc[0,6]
r_e=df_e.iloc[-1,6]/df_e.iloc[0,6]
r_f=df_f.iloc[-1,6]/df_f.iloc[0,6]
r_h=df_h.iloc[-1,6]/df_h.iloc[0,6]
r_g=df_g.iloc[-1,6]/df_g.iloc[0,6]
r_i=df_i.iloc[-1,6]/df_i.iloc[0,6]

print('202301:',r_a)
print('202302:',r_b)
print('202303:',r_c)
print('202304:',r_d)
print('202305:',r_e)
print('202306:',r_f)
print('202307:',r_h)
print('202308:',r_g)
print('202309:',r_i)


df_aa = pd.DataFrame({
    'col1': df_a.iloc[::60, 1],
    'col2': df_a.iloc[::60, 6]
})

df_bb = pd.DataFrame({
    'col1': df_b.iloc[::60, 1],
    'col2': df_b.iloc[::60, 6] * r_a
})

df_cc = pd.DataFrame({
    'col1': df_c.iloc[::60, 1],
    'col2': df_c.iloc[::60, 6] * r_a * r_b
})

df_dd = pd.DataFrame({
    'col1': df_d.iloc[::60, 1],
    'col2': df_d.iloc[::60, 6] * r_a * r_b * r_c
})
df_ee = pd.DataFrame({
    'col1': df_e.iloc[::60, 1],
    'col2': df_e.iloc[::60, 6] * r_a * r_b * r_c* r_d
})
df_ff = pd.DataFrame({
    'col1': df_f.iloc[::60, 1],
    'col2': df_f.iloc[::60, 6] * r_a * r_b * r_c* r_d * r_e
})
df_hh = pd.DataFrame({
    'col1': df_h.iloc[::60, 1],
    'col2': df_h.iloc[::60, 6] * r_a * r_b * r_c* r_d * r_e * r_f
})
df_gg = pd.DataFrame({
    'col1': df_g.iloc[::60, 1],
    'col2': df_g.iloc[::60, 6] * r_a * r_b * r_c* r_d * r_e * r_f * r_h
})

df_ii = pd.DataFrame({
    'col1': df_i.iloc[::60, 1],
    'col2': df_i.iloc[::60, 6] * r_a * r_b * r_c* r_d * r_e * r_f * r_h * r_g
})
# dataframe1 = pd.concat([df_aa, df_bb,
#                         df_cc,df_dd,df_ee
#                         #df_dd
#                         ], axis=0)
dataframe1 = pd.concat([df_aa, df_bb,
                        df_cc,df_dd,df_ee,df_ff,df_hh,df_gg, df_ii
                        #df_dd
                        ], axis=0)
# dataframe2 = dataframe1.iloc[::60, [1, 6]]
#
dataframe1 = dataframe1.reset_index(drop=True)
dataframe1['datetime'] = pd.to_datetime(dataframe1['col1']+2880000, unit='ms')
dataframe1 = dataframe1.set_index('datetime')
dataframe1[['col2']].plot()
plt.show()


daily_returns = dataframe1['col2'].resample('D').last().pct_change()
non_zero_returns = daily_returns[daily_returns != 0]
sharpe_ratio = non_zero_returns.mean() / non_zero_returns.std()

dataframe1.to_csv("/home/xianglake/yiliang/data/pnl_eosusdt_3.csv", encoding='gbk')


dataframe1['daily_return'] = dataframe1['col2'].pct_change()
mdd = max_drawdown(dataframe1['daily_return'])
final_proft = dataframe1['col2'].iloc[-1]/dataframe1['col2'].iloc[0]
proft_mdd_ratio = (final_proft-1)/abs(mdd)
print('------------------  Trading Result:             --------------------')
print('------------------  Profit:   '+str(final_proft)+' --------------------')
print('------------------  Maximum Drawdown:   '+str(mdd)+' --------------------')
print('------------------  Sharpe  Ratio:   '+str(sharpe_ratio)+' --------------------')
print('------------------  Profit Drawdown Ratio:   '+str(proft_mdd_ratio)+' --------------------')


dataframe1.drop('daily_return', axis=1, inplace=True)







#
# dataframe1.iloc[:, 2].plot()
#
# plt.show()



