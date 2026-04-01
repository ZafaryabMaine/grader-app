
read -p "Enter commit message (leave empty for default): " user_msg
if [ -z "$user_msg" ]; then
  user_msg="update"
fi
git add .
git commit -m "$user_msg"
git push