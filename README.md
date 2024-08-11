# movie_stuff
A Shiny app to visualize differences in multivariate test-retest reliability of FC between movie-watching and resting-state.

## To deploy to shinyapps.io
- In a terminal with python3.10, make sure all dependencies are installed by running the following code in the movie_stuff directory
```
python3 -m pip install -r requirements.txt
```
- Then to deploy with rsconnect,
```
cd ..
rsconnect deploy shiny movie_stuff --name vanderlab --title movie_stuff
```

### Troubleshooting:
shinyapps.io reported an error (calling /v1/applications/...): Not Found
Error: shinyapps.io reported an error (calling /v1/applications/...): Not Found
  - solution: delete the rsconnect-python directory from the movie_stuff directory. This gets made when you deploy, so if re-deploying then it can interfere sometimes.

