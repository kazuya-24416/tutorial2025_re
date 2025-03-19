sudo -s -- <<HERE
sed -i '/SSO SSH Config START/Q' /etc/ssh/sshd_config
echo "Port 3333" >> /etc/ssh/sshd_config
ssh-keygen -A
HERE

sudo mkdir -p /run/sshd

sudo /usr/sbin/sshd

git config --global user.email "kazukane0109@gmail.com"
git config --global user.name "Kazuya"

(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
    && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y