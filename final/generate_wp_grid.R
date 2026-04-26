# generate_wp_grid.R
# ==================
# Calls nflfastR::calculate_win_probability() to build a WP lookup grid.
# Scenario: trailing team just received kickoff after PAT, 1st & 10, own 30.
#
# Install: install.packages(c("nflfastR", "dplyr"))
# Run:     Rscript generate_wp_grid.R
# Output:  wp_grid.csv

library(nflfastR)
library(dplyr)

cat("Generating WP grid...\n")

# The trailing team has the ball (they received the kickoff after the PAT).
# posteam = trailing team. score_differential = -margin (they're behind).

grid <- expand.grid(
  margin = c(7, 8, 9),
  game_seconds_remaining = seq(0, 900, by = 30),
  trailing_timeouts = 0:3,
  leading_timeouts = 0:3,
  trailing_is_home = c(TRUE, FALSE),
  stringsAsFactors = FALSE
) %>% as_tibble()

grid <- grid %>%
  mutate(
    score_differential = -margin,
    down = 1,
    ydstogo = 10,
    yardline_100 = 70,  # own 30
    half_seconds_remaining = game_seconds_remaining,
    receive_2h_ko = 0,
    spread_line = 0,
    posteam_timeouts_remaining = trailing_timeouts,
    defteam_timeouts_remaining = leading_timeouts,

    # home_team / posteam determine the home flag inside the model
    # posteam is always the trailing team
    # if trailing team is home: posteam == home_team
    # if trailing team is away: posteam != home_team
    home_team = ifelse(trailing_is_home, "TRAIL", "LEAD"),
    posteam = "TRAIL"
  )

cat(sprintf("  Grid: %d rows\n", nrow(grid)))
cat("  Calling nflfastR::calculate_win_probability()...\n")

result <- nflfastR::calculate_win_probability(grid)

result <- result %>%
  mutate(
    trailing_wp = wp,
    leading_wp = 1 - wp
  )

output <- result %>%
  select(
    margin,
    game_seconds_remaining,
    trailing_timeouts,
    leading_timeouts,
    trailing_is_home,
    trailing_wp,
    leading_wp
  )

write.csv(output, "wp_grid.csv", row.names = FALSE)
cat(sprintf("  Saved wp_grid.csv (%d rows)\n\n", nrow(output)))

# Sanity checks
cat("Sanity checks (5 min, 3v3 TOs):\n")
for (h in c(TRUE, FALSE)) {
  label <- ifelse(h, "Trailing team HOME", "Trailing team AWAY")
  cat(sprintf("\n  %s:\n", label))
  for (m in c(7, 8, 9)) {
    row <- output %>%
      filter(margin == m, game_seconds_remaining == 300,
             trailing_timeouts == 3, leading_timeouts == 3,
             trailing_is_home == h)
    cat(sprintf("    Down %d: trailing WP = %.4f, leading WP = %.4f\n",
                m, row$trailing_wp, row$leading_wp))
  }
}

cat("\nDone! Now run: python process_data.py\n")
