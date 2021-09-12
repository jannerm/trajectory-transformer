export DOWNLOAD_PATH=logs

[ ! -d ${DOWNLOAD_PATH} ] && mkdir ${DOWNLOAD_PATH}

## downloads pretrained models for 16 datasets:
	## {halfcheetah, hopper, walker2d, ant}
	## 		x
	## {expert-v2, medium-expert-v2, medium-v2, medium-replay-v2}

wget https://www.dropbox.com/sh/r09lkdoj66kx43w/AACbXjMhcI6YNsn1qU4LParja?dl=1 -O dropbox_models.zip
unzip dropbox_models.zip -d ${DOWNLOAD_PATH}
rm dropbox_models.zip

## downloads 15 plans from each pretrained model
wget https://www.dropbox.com/s/5sn79ep79yo22kv/pretrained-plans.tar?dl=1 -O dropbox_plans.tar
tar -xvf dropbox_plans.tar
cp -r pretrained-plans/* ${DOWNLOAD_PATH}
rm -r pretrained-plans
rm dropbox_plans.tar
