## AZURE_STORAGE_CONNECTION_STRING has a substring formatted lik:
	## AccountName=${STORAGE_ACCOUNT};AccountKey=${STORAGE_KEY};EndpointSuffix= ...
export AZURE_STORAGE_ACCOUNT=`(echo $AZURE_STORAGE_CONNECTION_STRING | grep -o -P '(?<=AccountName=).*(?=;AccountKey)')`
export AZURE_STORAGE_KEY=`(echo $AZURE_STORAGE_CONNECTION_STRING | grep -o -P '(?<=AccountKey=).*(?=;EndpointSuffix)')`

echo "accountName" ${AZURE_STORAGE_ACCOUNT} > ./azure/fuse.cfg
echo "accountKey" ${AZURE_STORAGE_KEY} >> ./azure/fuse.cfg
echo "containerName" ${AZURE_STORAGE_CONTAINER} >> ./azure/fuse.cfg