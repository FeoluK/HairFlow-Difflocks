#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


# username and password input
echo -e "\nYou need to register at https://difflocks.is.tue.mpg.de/"
read -p "Username (DIFFLOCKS):" username
read -p "Password (DIFFLOCKS):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading files..."


#checkpoints
wget --post-data "username=$username&password=$password" \
  "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=difflocks_checkpoints.zip" \
  -O "difflocks_checkpoints.zip" --no-check-certificate --continue
# Check if wget failed (non-zero exit code)
if [ $? -ne 0 ]; then
    echo "❌ Error downloading body data. Exiting script."
    exit 1
fi

#unzip
unzip difflocks_checkpoints.zip 


