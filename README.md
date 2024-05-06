# DistVec
This is the source code to the paper "DistVec: A Plugin Toolkit for Efficient Distributed Machine Learning in Parallel Database Systems". Please refer to the paper for the experimental details.
# Table of Content
1. Environment Preparation
2. DataSet Preparation
3. System & Database config
4. Running example
# Environment Preparation
Firstly, we use the Centos7 system to build a distributed database environment.
For information on Centos7, please refer to the official website and documentation.

https://www.centos.org/

Regarding distributed databases, we use greenplum 6.20.0. 
Please install the greenplum database on the prepared node cluster and refer to the official documentation.

https://docs.vmware.com/en/VMware-Greenplum/6/greenplum-database/install_guide-install_guide.html

Finally, after preparing the environment, you also need to install madlib=1.19.0 in your cluster. 

The official documentation can be referred to.https://madlib.apache.org/docs/v1.19.0/index.html

Attention: Please try to avoid installing all nodes on the same machine, which may cause communication failure.
# DataSet Preparation
In our paper, we used a total of three datasets, namely criteo, imagenet, and ogbn mag. The corresponding download addresses and their preprocessing scripts are all located in the /dataset folder in the root directory.

After processing the dataset into CSV format and storing it through a preprocessing script, you can read the data into the database in the following form, using Criteo as an example.

`create table criteo(id int, xi numeric[], xv numeric[], y int);`

`copy criteo from '/your_dataset_path.csv' with CSV HEADER;`

Afterwards, Madlib's data compression UDF can be used for compression storage.

```
select madlib.training_preprocessor_dl(source_table,
                         output_table,
                         dependent_varname,
                         independent_varname,
                         buffer_size,
                         normalizing_const,
                         num_classes,
                         distribution_rules
                        )
```
# System & Database config
Based on our running tests, you can make modifications according to the specific situation of your environment.

```
[System config]
SHMMAX = $(getconf _PHYS_PAGES)/2
SHMALL = $(getconf _PHYS_PAGES)/2 * $(getconf PAGE_SIZE)
vm.overcommit_memory = 2
vm.overcommit_ratio = 95

net.ipv4.ip_local_port_range = 10000 65535 # See Port Settings
kernel.sem = 250 2048000 200 8192
kernel.sysrq = 1
kernel.core_uses_pid = 1
kernel.msgmnb = 65536
kernel.msgmax = 65536
kernel.msgmni = 2048
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.tcp_max_syn_backlog = 4096
net.ipv4.conf.all.arp_filter = 1
net.ipv4.ipfrag_high_thresh = 41943040
net.ipv4.ipfrag_low_thresh = 31457280
net.ipv4.ipfrag_time = 60
net.core.netdev_max_backlog = 10000
net.core.rmem_max = 2097152
net.core.wmem_max = 2097152
vm.swappiness = 10
vm.zone_reclaim_mode = 0
vm.dirty_expire_centisecs = 500
vm.dirty_writeback_centisecs = 100
vm.dirty_background_ratio = 0 # See System Memory
vm.dirty_ratio = 0
vm.dirty_background_bytes = 1610612736
vm.dirty_bytes = 4294967296
```

```
[Database config]
change example:
  gpconfig -s shared_buffers #show
  gpconfig -c shared_buffers -v 8GB # set

shared_buffers = 1GB
work_mem = 256MB
temp_buffers = 128MB
max_statement_mem = 2GB
gp_enable_gpperfmon = off
gp_resqueue_memory_policy = eager_free
gp_vmem_protect_limit = 16384
max_connections = 2500
```

```
[Gpu config]
  cuda = 10.0
  cudnn = 7.6.5
  tensorflow-gpu = 1.15.0
```
