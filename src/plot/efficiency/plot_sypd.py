import matplotlib.pyplot as plt
import numpy as np

# Values represent the time (in seconds) needed to simulate 1 month. They are the "Elapsed time" 
# read from the job.out of the third month of the simulation.
# [0] = GOSI10p0 1/4  u-dd982
# [1] = GOSI10p0 1/4 AGRIF GS zoom 1/12 u-dd748
# [2] = GOSI9    1/4  u-ct401
# [3] = GOSI9    1/12 u-ct406
#time = np.asarray([2504., 4418., 3175., 2196.])

# [0] = GOSI10p0 1/4  u-dd982
# [1] = GOSI10p0 1/4 AGRIF GS zoom 1/12 u-dd748
# [3] = GOSI10p0 1/12 u-de576
time = np.asarray([2504., 4418., 4301.])

sypd = (1. / 12.) / (time / (60.*60.*24.))
#color = ['blue','green','deepskyblue', 'red']
color = ['blue','green','red']

#x = np.arange(4)
x = np.arange(3)
r=plt.bar(x, height=sypd, color=color)
#plt.xticks(x, ['GOSI10-1/4','GOSI10-1/4 \nAGRIF-1/12','GOSI9-1/4','GOSI9-1/12'])
plt.xticks(x, ['GOSI10-1/4','GOSI10-1/4 \nAGRIF-1/12','GOSI10-1/12'])
plt.ylabel('Simulated Years Per Day [SYPD]')
plt.bar_label(r, padding=3, fmt='%.2f', fontsize='small')

plt.savefig("sypd_talk.png", bbox_inches="tight")
print("done")
plt.close()

