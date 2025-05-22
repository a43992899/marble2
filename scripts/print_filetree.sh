find . \
  \( -type d \( -name .git -o -name 'marble.egg-info' \) \) -prune \
  -o -print \
| sed \
  -e 's/[^-][^\/]*\//|   /g' \
  -e 's/|   \([^|]\)/|── \1/'
