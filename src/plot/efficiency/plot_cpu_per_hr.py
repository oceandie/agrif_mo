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

# Values represent the number of processors allocated to NEMO
# from NEMO_NPROC variable in rose_suite.conf
#proc = np.asarray([345., 345., 344., 6150.])
nemo_proc = np.asarray([345., 345., 6149.])
# Values represent the number of processors allocated to XIOS
# from XIOS_NPROC variable in rose_suite.conf
xios_proc = np.asarray([8., 8., 72.])

PrHr = (nemo_proc+xios_proc) * (time / (60.*60.))

#color = ['blue','green','deepskyblue', 'red']
color = ['blue','green','red']

#x = np.arange(4)
x = np.arange(3)
r = plt.bar(x, height=PrHr, color=color)
#plt.xticks(x, ['GOSI10-1/4','GOSI10-1/4 \nAGRIF-1/12','GOSI9-1/4','GOSI9-1/12'])
plt.xticks(x, ['GOSI10-1/4','GOSI10-1/4 \nAGRIF-1/12','GOSI10-1/12'])
plt.ylabel('CPU Hours [hours]')
plt.bar_label(r, padding=3, fmt='%.2f', fontsize='small')

plt.savefig("cpuhr_talk.png", bbox_inches="tight")
print("done")
plt.close()

