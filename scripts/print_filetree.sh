find . \
  \( -type d \( -name .git -o -name 'marble.egg-info' -o -name '__pycache__' \) \) -prune \
  -o -print \
| sed \
  -e 's/[^-][^\/]*\//|   /g' \
  -e 's/|   \([^|]\)/|── \1/'
