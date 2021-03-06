#!/bin/bash -x

# Push to GitHub
if [ -n "$GITHUB_API_KEY" ]; then
  print "pushing to github"
  cd $TRAVIS_BUILD_DIR
  git checkout --orphan pdf
  git rm -rf .
  git add -f paper/uberchemical.pdf
  git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
  git push -q -f https://andycasey:$GITHUB_API_KEY@github.com/andycasey/uber-chemi-cal pdf
fi
