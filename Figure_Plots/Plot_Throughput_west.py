# center of HKUST Campus
# This code is used to plot the throughput and idle rate of different methods
import matplotlib.pyplot as plt

x = [0, 5, 10, 15, 20, 25, 30]
Y_rule_thro = [0,410.06,869.68,991.41,979.47,964.44,972.78]
Y_llm_thro =  [0, 44.67,318.46,578.26,715.76,725.62,737.12]
Y_agent_thro = [0,404.06,767.4,866.29,912.93,905.34,923.56]

# idle rate
Y_rule_idle = [100 , 48.46, 16.15, 3.53, 0.45, 0.0, 0.0]
Y_llm_idle =   [100, 83.08, 56.92 , 26.54,9.62,4.23,0.38]
Y_agent_idle = [100,49.31,19.38,8.69,2.69,1.31,0.0]

# plot
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# color map
COLOR_MAP = {
    'Rule-based': '#91CAE8',
    'Prompt-based': '#91CAE8',
    'WirelessAgent': '#91CAE8'
}


ax1.plot(x, Y_rule_thro, color=COLOR_MAP['Rule-based'], linestyle='-', 
        linewidth=2, marker='o', markersize=9, label='_nolegend_')
ax1.plot(x, Y_agent_thro, color=COLOR_MAP['WirelessAgent'], linestyle='--',
        linewidth=2, marker='^', markersize=9, label='_nolegend_')
ax1.plot(x, Y_llm_thro, color=COLOR_MAP['Prompt-based'], linestyle=':',
        linewidth=2, marker='s', markersize=9, label='_nolegend_')

# color map
COLOR_MAP = {
    'Rule-based': '#333333',
    'Prompt-based': '#333333',
    'WirelessAgent': '#333333'
}

ax2.plot(x, Y_rule_idle, color=COLOR_MAP['Rule-based'], linestyle='-',
        linewidth=2, marker='o', markersize=9, label='Rule-based')
ax2.plot(x, Y_agent_idle, color=COLOR_MAP['WirelessAgent'], linestyle='--',
        linewidth=2, marker='^', markersize=9, label='WirelessAgent')
ax2.plot(x, Y_llm_idle, color=COLOR_MAP['Prompt-based'], linestyle=':',
        linewidth=2, marker='s', markersize=9, label='Prompt-based')


ax1.set_xlabel('Number of Users', fontsize=20)
ax1.set_ylabel('Total Throughput (Mbps)', color='#91CAE8', fontsize=20)
ax2.set_ylabel('Bandwidth Idle Rate (%)', fontsize=20)


handles, labels = ax2.get_legend_handles_labels()
legend = ax1.legend(handles, labels,
                  loc='upper left',
                  bbox_to_anchor=(0.65, 4/10),
                  frameon=True,
                  fontsize=12,
                  ncol=1,
                  borderpad=0.5,
                  handletextpad=0.5,
                  labelspacing=0.5)



ax1.add_artist(legend)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax1.set_xlim(0, 31)
ax1.grid(True, linestyle=':', alpha=0.7)
ax2.set_ylim(0, 100)
plt.tight_layout()

output_path = r'F:\code\draw\picture\Plot_Throughput_west.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Figure saved to {output_path}')

plt.show()