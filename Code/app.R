library(shiny)
library(shinyvalidate)
library(shinyjs)
library(tidyverse)
library(WVPlots)
library(rsconnect)
#library(reticulate)
library(VGAM)
library(tidyr)
library(stringr)
library(splitTools)
library(RColorBrewer)
library(plotly)
library(DT)



#---------------------------------- Read Data -----------------------------------#
CA_Asian_business_review = read.csv('CA_Asian_business_review.csv')
data = read.csv("final_data.csv")






#---------------------------------- Modeling -----------------------------------#


Y = data["comment_star"]
factorY = factor(c(t(Y)))
data$OrderStar = ordered(factorY,labels=c("1","2","3","4","5"))
col_list = c(colnames(data))
data[,c(1,6:16,18,19,20)] <- lapply(data[,c(1,6:16,18,19,20)], factor)


data_china=subset.data.frame(data, data$general_category == "Chinese" & data$Total_hour!=0)
data_jpn=subset.data.frame(data, data$general_category == "Japanese" & data$Total_hour!=0)
data_krn=subset.data.frame(data, data$general_category == "Korean" & data$Total_hour!=0)
data_AF=subset.data.frame(data, data$general_category == "Asian Fusion" & data$Total_hour!=0)

train.rows <- sample(rownames(data), dim(data)[1]*0.8)
general_train <- data[train.rows,]



train_test_split = function(df){
  train.rows <- sample(rownames(df), dim(df)[1]*0.8)
  train <- df[train.rows,]
  return(train)
}


CN_model  = vglm(OrderStar ~  BikeParking 
                 + HasTV 
                 + Alcohol
                 + lot
                 + dinner 
                 + Total_hour 
                 + upscale_classy
                 ,family = cumulative(parallel = TRUE),data = train_test_split(data_china))

JP_model = vglm(OrderStar ~  BikeParking 
                + RestaurantsGoodForGroups 
                + WiFi 
                + garage 
                + street 
                + Total_hour 
                + upscale_classy,
                family = cumulative(parallel = TRUE),data = train_test_split(data_jpn))

AF_model = vglm(OrderStar ~  BikeParking 
                + RestaurantsReservations 
                + HasTV 
                + Alcohol
                + RestaurantsGoodForGroups 
                + WiFi 
                + lot
                + valet 
                + dinner 
                + Total_hour 
                + upscale_classy,
                family = cumulative(parallel = TRUE),data = train_test_split(data_AF))

general_model = vglm(OrderStar ~  BikeParking 
                     + RestaurantsReservations 
                     + Alcohol
                     + WiFi 
                     + garage 
                     + street
                     + lot
                     + valet 
                     + dinner 
                     + Total_hour 
                     + acceptable_noise 
                     + upscale_classy,family = cumulative(parallel = TRUE),data = general_train)












#---------------------- Model prediction pie chart --------------------#

predict_model = function(category, BikeParking,RestaurantsGoodForGroups, WiFi,garage,valet, street,Total_hour,acceptable_noise,upscale_classy, HasTV, dinner, RestaurantsReservations, lot,Alcohol){
  columns = c('OrderStar','comment_star','BikeParking','RestaurantsGoodForGroups','WiFi','garage','valet', 'street_parking','Total_hour','acceptable_noise','upscale_classy','HasTV','dinner','RestaurantsReservations','lot','Alcohol')
  test_df = data.frame(matrix(nrow = 0, ncol = length(columns)))
  colnames(test_df) = columns
  
  
  new_row = list('BikeParking' = BikeParking,'RestaurantsGoodForGroups' = RestaurantsGoodForGroups,'WiFi' = WiFi,
                 'garage' = garage,'valet' = valet, 'street_parking' = street,
                 'Total_hour' = Total_hour,'acceptable_noise' = acceptable_noise,
                 'upscale_classy' = upscale_classy,'HasTV' = HasTV, 'dinner' = dinner,
                 'RestaurantsReservations' = RestaurantsReservations,
                 'lot' = lot,'Alcohol' = Alcohol)
  
  
  test_df = rbind(test_df,new_row)
  test_df$BikeParking = as.factor(test_df$BikeParking)
  test_df$RestaurantsGoodForGroups = as.factor(test_df$RestaurantsGoodForGroups)
  test_df$WiFi = as.factor(test_df$WiFi)
  test_df$garage = as.factor(test_df$garage)
  test_df$valet = as.factor(test_df$valet)
  test_df$street = as.factor(test_df$street)
  test_df$acceptable_noise = as.factor(test_df$acceptable_noise)
  test_df$upscale_classy = as.factor(test_df$upscale_classy)
  test_df$HasTV = as.factor(test_df$HasTV)
  test_df$dinner = as.factor(test_df$dinner)
  test_df$RestaurantsReservations = as.factor(test_df$RestaurantsReservations)
  test_df$lot = as.factor(test_df$lot)
  test_df$Alcohol = as.factor(test_df$Alcohol)
  
  
  
  #remove columns based on different categories
  if (category == 'Chinese'){
    test_df = subset(test_df, select = c('BikeParking', 'HasTV','Alcohol','lot','dinner',
                                         'Total_hour','upscale_classy'))
    model = CN_model
  }
  else if(category == 'Japanese'){
    test_df = subset(test_df, select = c('BikeParking',
                                         'RestaurantsGoodForGroups','WiFi','garage',
                                         'street','Total_hour','upscale_classy'))
    model = JP_model
  }
  # else if(category == 'Korean'){
  #   
  # }
  else if (category == 'Asian Fusion'){
    test_df = subset(test_df, select = c('BikeParking',
                                         'RestaurantsGoodForGroups','WiFi','HasTV',
                                         'Alcohol','RestaurantsReservations',
                                         'lot','valet','dinner','Total_hour','upscale_classy'))
    model = AF_model
  }
  else{
    test_df = test_df
    model = general_model
  }
  
  
  pred <- predict(model,test_df,type = "response")
  pie.prop <- as.vector(pred)
  pie.labels <- c("1 star","2 star", "3 star", "4 star","5 star")
  pie.labels <- paste(pie.labels," ",round(pie.prop,2) , sep="")
  return(pie(pie.prop, pie.labels, col=brewer.pal(5, "RdYlGn")))
}



#------------------------------------ Model interpretation --------------------------------------------#
model_intp = function(category, BikeParking,RestaurantsGoodForGroups, WiFi,garage,valet, street,Total_hour,acceptable_noise,upscale_classy, HasTV, dinner, RestaurantsReservations, lot, Alcohol){
  columns = c('OrderStar','comment_star','BikeParking','RestaurantsGoodForGroups','WiFi','garage','valet', 'street_parking','Total_hour','acceptable_noise','upscale_classy','HasTV','dinner','RestaurantsReservations','lot','Alcohol')
  test_df = data.frame(matrix(nrow = 0, ncol = length(columns)))
  colnames(test_df) = columns
  
  
  new_row = list('BikeParking' = BikeParking,'RestaurantsGoodForGroups' = RestaurantsGoodForGroups,'WiFi' = WiFi,
                 'garage' = garage,'valet' = valet, 'street_parking' = street,
                 'Total_hour' = Total_hour,'acceptable_noise' = acceptable_noise,
                 'upscale_classy' = upscale_classy,'HasTV' = HasTV, 'dinner' = dinner,
                 'RestaurantsReservations' = RestaurantsReservations,
                 'lot' = lot,'Alcohol' = Alcohol)
  
  
  test_df = rbind(test_df,new_row)
  test_df$BikeParking = as.factor(test_df$BikeParking)
  test_df$RestaurantsGoodForGroups = as.factor(test_df$RestaurantsGoodForGroups)
  test_df$WiFi = as.factor(test_df$WiFi)
  test_df$garage = as.factor(test_df$garage)
  test_df$valet = as.factor(test_df$valet)
  test_df$street = as.factor(test_df$street)
  test_df$acceptable_noise = as.factor(test_df$acceptable_noise)
  test_df$upscale_classy = as.factor(test_df$upscale_classy)
  test_df$HasTV = as.factor(test_df$HasTV)
  test_df$dinner = as.factor(test_df$dinner)
  test_df$RestaurantsReservations = as.factor(test_df$RestaurantsReservations)
  test_df$lot = as.factor(test_df$lot)
  test_df$Alcohol = as.factor(test_df$Alcohol)
  
  
  
  #remove columns based on different categories
  if (category == 'Chinese'){
    test_df = subset(test_df, select = c('BikeParking', 'HasTV','Alcohol','lot','dinner',
                                         'Total_hour','upscale_classy'))
    model = CN_model
  }
  else if(category == 'Japanese'){
    test_df = subset(test_df, select = c('BikeParking',
                                         'RestaurantsGoodForGroups','WiFi','garage',
                                         'street','Total_hour','upscale_classy'))
    model = JP_model
  }
  # else if(category == 'Korean'){
  #   
  # }
  else if (category == 'Asian Fusion'){
    test_df = subset(test_df, select = c('BikeParking',
                                         'RestaurantsGoodForGroups','WiFi','HasTV',
                                         'Alcohol','RestaurantsReservations',
                                         'lot','valet','dinner','Total_hour','upscale_classy'))
    model = AF_model
  }
  else{
    test_df = test_df
    model = general_model
  }
  
  
  pred <- predict(model,test_df,type = "response")
  return(pred)
}
  













#-------------------------------------------------- Shiny app ----------------------------------------------------#

ui = fluidPage(

  titlePanel('Your customer review and satisfication analysis'),
  column(2,
         wellPanel(
           titlePanel('Restaurant name'),
           selectInput('category', 'Select business category', choice = unique(CA_Asian_business_review$general_category)),
           uiOutput('business_name'))
  ),
  
  mainPanel(
    tabsetPanel(

      tabPanel('Text Frequency Table',
               textOutput('ngram_hint'),
               tags$head(tags$style("#ngram_hint{color: red;
     font-size: 15px;font-style: bold;}")),
               br(),
               br(),
               column(3,numericInput('n_result','Top N view', value = 1, min = 1, max = 100)),
               column(3, numericInput('n_gram','length of text', value = 1, min = 1, max = 10)),
               column(3, selectInput('sentiment','Choose a sentiment', choice = unique(CA_Asian_business_review$sentiment))),
               dataTableOutput('n_gram_table')
      ),
      
      tabPanel('Text Importance Table',
               textOutput('tfidf_hint'),
               tags$head(tags$style("#tfidf_hint{color: red;
     font-size: 15px;font-style: bold;}")),
               br(),
               br(),
               column(3,numericInput('TI_N_result','Top N view', value = 1, min = 1, max = 50)),
               column(3,numericInput('TI_n_gram','length of text', value = 1, min = 1, max = 10)),
               column(3,selectInput('TI_sentiment','Choose a sentiment', choice = unique(CA_Asian_business_review$sentiment))),
               dataTableOutput('text_importance_table')
      ),
      
      tabPanel('Parking map', includeHTML('parking.html')),
      tabPanel('Feature modeling',
               textOutput('modeling_instruction'),
               tags$head(tags$style("#modeling_instruction{color: blue;
       font-size: 15px;font-style: bold;}")),
               br(),
               br(),
               
               column(3, selectInput('wifi','WIFI', choices =list(0,1)),
                      numericInput('total_hour','Total_opening_hour',value = 50,min = 1,max=168),
                      selectInput('lot','Parking lot', choices = list(0,1)),
               ), 
               
               column(3, selectInput('valet','Valet Parking', choices = list(0,1)),
                      selectInput('garage','Garage Parking', choices = list(0,1)),
                      selectInput('street_park','Street Parking', choices = list(0,1)),
               ),
               
               column(3, selectInput('noise','Noise level', choices = list(0,1)),
                      selectInput('upscale_classy','Upscale or Classy', choices = list(0,1)),
                      selectInput('bike_park','Bike Parking', choices = list(0,1)),
                      selectInput('alcohol','Alcohol', choices = list(0,1))
               ),
               
               column(3, selectInput('TV', 'Has TV', choices = list(0,1)),
                      selectInput('dinner','Dinner provided', choices = list(0,1)),
                      selectInput('good_for_group','Group friendly', choices = list(0,1)),
                      selectInput('reservation','Reservation provided', choices = list(0,1))),
               
               
               fluidPage(
                 column(6, plotOutput('modeling')),
                 column(6,
                        br(),
                        br(),
                        br(),
                        br(),
                        br(),
                        htmlOutput('model_interpretation'),
                        tags$head(tags$style("#model_interpretation{color: blue;
               font-size: 13px;font-style: bold;}"))
                 )
               ),
               textOutput('modeling_hint'),
               tags$head(tags$style("#modeling_instruction{color: red;font-size: 13px;
        font-style: bold;
        }")),
      )
    )
  )
  
)



# ------------------ App server logic (Edit anything below) --------------- #
server = function(input, output){
  
  #---------------------Initialize virtual env-----------------------#
  virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
  python_path = Sys.getenv('PYTHON_PATH')
  PYTHON_DEPENDENCIES = c('pandas','numpy','plotly','matplotlib','scikit-learn','tqdm','chardet','seaborn','nltk','textblob','wordcloud')

  # Create virtual env and install dependencies
  reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
  reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, ignore_installed=TRUE)
  reticulate::use_virtualenv(virtualenv_dir, required = T)
  
  
  #Apply python function
  reticulate::source_python('py_fun.py')
  

  
  output$business_name = renderUI({
    if (input_provided(input$category)){
      cur_df = CA_Asian_business_review[CA_Asian_business_review['general_category']== input$category, ]
      selectInput('business_name','Select business name', choice = unique(cur_df$name))
    }
    else {
      selectInput('business_name','Select business name', choice = unique(CA_Asian_business_review$name))
    }
  })
  
  
  output$n_gram_table = renderDataTable({
    n_gram_table(input$n_gram, input$business_name, input$n_result,input$sentiment)
  })
  
  output$text_importance_table = renderDataTable({
    tfidf_table(input$business_name,input$TI_sentiment, input$TI_n_gram, input$TI_N_result, input$category)
  })

  
  
  output$modeling = renderPlot({
    if(input_provided(input$category) == FALSE){
      category = 'other'
    }else{
      if (input$category == 'Chinese'){
        category ='Chinese'
      }
      else if (input$category == 'Japanese'){
        category ='Chinese'
      }
      else if (input$category == 'Korean'){
        category ='Korean'
      }
      else {
        category ='Asian Fusion'
      }
    }
    predict_model(category,input$bike_park,input$good_for_group,input$wifi,input$garage,input$valet, input$street_park,input$total_hour,input$noise,input$upscale_classy, input$TV, input$dinner, input$reservation, input$lot, input$alcohol)
  })
  
  
  
  
  #------------------ Instruction text ------------------#
  output$wordCloud_instruction = renderText({
    'Hint: the more a specific word appears in a source of textual data, the bigger and bolder it appears in the word cloud'
  })
  
  
  output$modeling_hint = renderText({
    'Hint: 0 means you do not have the feature, 1 means you want this feature'
  })
  
  
  output$modeling_instruction = renderText({
    'Instruction: This part is only related to category & features appeared in the page!!!'
  })
  
  
  output$ngram_hint = renderText({
    'Hint: the higher the count, the more frequent a word/text appears'
  })
  
  output$tfidf_hint = renderText({
    'Hint: the higher the TF-IDF score, the more important or relevant to term is'
  })
  
  
  output$model_interpretation = renderText({
    if(input_provided(input$category) == FALSE){
      category = 'other'
    }else{
      if (input$category == 'Chinese'){
        category ='Chinese'
      }
      else if (input$category == 'Japanese'){
        category ='Chinese'
      }
      else if (input$category == 'Korean'){
        category ='Korean'
      }
      else {
        category ='Asian Fusion'
      }
    }
    pred = model_intp(category,input$bike_park,input$good_for_group,input$wifi,input$garage,input$valet, input$street_park,input$total_hour,input$noise,input$upscale_classy, input$TV, input$dinner, input$reservation, input$lot, input$alcohol)
    str1 = paste0("The predicted probability of getting star 1 is: ", 100*round(pred[1],2), '%')
    str2 = paste0("The predicted probability of getting star 2 is: ", 100*round(pred[2],2), '%')
    str3 = paste0("The predicted probability of getting star 3 is: ", 100*round(pred[3],2), '%')
    str4 = paste0("The predicted probability of getting star 4 is: ", 100*round(pred[4],2), '%')
    str5 = paste0("The predicted probability of getting star 5 is: ", 100*round(pred[5],2), '%')
    HTML(paste(str1, str2, str3,str4, str5, sep = '<br/>'))
  })
  
  
  
}

shinyApp(ui, server)







