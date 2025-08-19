#Processes and exports distribution plots for Figure 5


install.packages(c("ggridges", "ggplot2", "reshape2", "plyr", "dplyr", "tidyverse","plotly","GGaly","rio","ggthemes","foreach","parallel","doParallel","foreach","EnvStats","rstatix","psych"))
library(readxl)
library(reshape2) 
library(plyr)
library(dplyr)
library(ggthemes) # Charger
library(tidyverse)
library(bigmemory)
library(plotly)
library(GGally)
library(rio)
library(data.table)
library(ggridges)
library(doParallel)
library(remotes)

remotes::install_github('rpkgs/gg.layers')
library(gg.layers)
library(ggplot2)

devtools::install_github("associatedpress/aptheme")

library(aptheme)
library(psych)
library(rstatix)
library(EnvStats)


remotes::install_github("coolbutuseless/ggpattern")
library(readr)





# Total

dftotal <- read_csv("../ALlvaluesfordistrib_total.csv")



# attempt with one facet

install.packages("devtools")  # only if not already installed
devtools::install_github("clauswilke/ggridges", ref = "pattern")


dftotal <- dftotal %>%
  mutate(
    config = as.factor(config),
    IAM = as.factor(IAM),
    `Wheat market` = as.factor(`Wheat market`),
    `PV comparison` = as.factor(`PV comparison`),
    ssp_year = paste(ssp, year)
  )


dftotal <- dftotal %>%
  group_by(ssp_year) %>%
  mutate(median_value = median(value, na.rm = TRUE)) %>%
  ungroup()

#dftotal$ssp_year <- factor(dftotal$ssp_year, levels = dftotal %>%
#                             group_by(ssp_year) %>%
#                             summarise(med = median(value, na.rm = TRUE)) %>%
#                             arrange(med) %>%
#                             pull(ssp_year))

dftotal$`Wheat market` <- factor(
  dftotal$`Wheat market`,
  levels = c("local", setdiff(unique(dftotal$`Wheat market`), "local"))
)







p <-
  ggplot(dftotal, aes(
    x = value,
    y = ssp_year,
    fill = config,
    color = IAM,
    linetype = `PV comparison`
  )) +
  geom_density_ridges(
    alpha = 0.4,
    scale = 1.25,
    linewidth = 0.4,
    rel_min_height = 0.00001,
    
  ) +
  scale_x_continuous(
    limits = c(-0.6, 0.5),
    
    breaks = seq(-0.6, 0.5, by = 0.2),
    labels = scales::number_format(accuracy = 0.1)
  ) +
  

  facet_wrap(~ `Wheat market`, ncol = 2, scales = "free_y") +
  
  # Manual fill colors for config
  scale_fill_manual(
    name = "Config",
    values = c(
      "1" = "tan2",  # light teal
      "2" = "royalblue3"    # orange
    )
  ) +
  
  # Manual line colors for IAM
  scale_color_manual(
    name = "IAM",
    values = c(
      "image" = "red2",
      "remind" = "gray4"
    )
  ) +
  
  # Line types for PV comparison
  scale_linetype_manual(
    name = "PV comparison",
    values = c("ST" = "solid", "fixed" = "dashed")
  ) +
  
  labs(
    x = "Value",
    y = "SSP-Year",
    title = "Custom Ridgeplot"
  ) +
  theme_ridges() +
  theme(
    legend.position = "right",
    strip.text = element_text(size = 12, face = "bold")
  )

ggsave("my_plot2.jpg", plot = p, width = 10, height = 6, dpi = 900)




# crop main

dfcrop <- read_csv("../ALlvaluesfordistrib_cropmain.csv")




dfcrop <- dfcrop %>%
  mutate(
    config = as.factor(config),
    IAM = as.factor(IAM),
    ssp_year = paste(ssp, year)
  )


dfcrop <- dfcrop %>%
  group_by(ssp_year) %>%
  mutate(median_value = median(value, na.rm = TRUE)) %>%
  ungroup()



# refining scales

dfcrop$group <- ifelse(dfcrop$ssp_year %in% c("SSP1-Base 2050", "SSP1-Base 2070","SSP2-Base 2050","SSP2-Base 2070","SSP2-Base 2090"), "Large", "Small")

# Then facet by that group with free x-axis scale


# other attempt

# Example grouping
df_large <- dfcrop %>% filter(group == "Large")
df_small <- dfcrop %>% filter(group == "Small")

# Create plot for each group with different x limits
p_small <-  ggplot(df_small, aes(
  x = value,
  y = ssp_year,
  fill = config,
  color = IAM,
)) +
  geom_density_ridges(
    alpha = 0.4,
    scale = 1.5,
    linewidth = 0.4,
    rel_min_height = 0.00001,
    
  )  +
  scale_x_continuous(
    limits = c(-50, 10),
    
    breaks = seq(-50, 10, by = 5),
    labels = scales::number_format(accuracy = 1)
  ) +
  
  # Manual fill colors for config
  scale_fill_manual(
    name = "Config",
    values = c(
      "1" = "tan2",  # light teal
      "2" = "royalblue3"    # orange
    )
  ) +
  
  # Manual line colors for IAM
  scale_color_manual(
    name = "IAM",
    values = c(
      "image" = "red2",
      "remind" = "gray4"
    )
  ) +
  
  
  
  labs(
    x = "Value",
    y = "SSP-Year",
    title = "Custom Ridgeplot"
  ) +
  theme_ridges() +
  theme(
    legend.position = "right",
    strip.text = element_text(size = 12, face = "bold")
  )



p_large <-  ggplot(df_large, aes(
  x = value,
  y = ssp_year,
  fill = config,
  color = IAM,
)) +
  geom_density_ridges(
    alpha = 0.4,
    scale = 1.5,
    linewidth = 0.4,
    rel_min_height = 0.00001,
    
  )  +
  scale_x_continuous(
    limits = c(-700, 50),
    breaks = seq(-700, 100, by = 50),
    
    labels = scales::number_format(accuracy = 1)
  ) +
  
  # Manual fill colors for config
  scale_fill_manual(
    name = "Config",
    values = c(
      "1" = "tan2",  # light teal
      "2" = "royalblue3"    # orange
    )
  ) +
  
  # Manual line colors for IAM
  scale_color_manual(
    name = "IAM",
    values = c(
      "image" = "red2",
      "remind" = "gray4"
    )
  ) +
  
  
  
  labs(
    x = "Value",
    y = "SSP-Year",
    title = "Custom Ridgeplot"
  ) +
  theme_ridges() +
  theme(
    legend.position = "right",
    strip.text = element_text(size = 12, face = "bold")
  )

# Combine both plots vertically
library(patchwork)
pcombined<-p_large / p_small

# Save as PDF
ggsave("combined_plot_crop2.pdf", plot = pcombined, width = 10, height = 8, dpi = 900)

# Save as JPG
ggsave("combined_plotcrop2.jpg", plot = pcombined, width = 10, height = 8, dpi = 900)

















# 
# 
