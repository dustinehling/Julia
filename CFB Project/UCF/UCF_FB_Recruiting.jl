#Libraries
using CategoricalArrays
using Clustering
using CSV
using Dates
using DataFrames
using DecisionTree
using Distributions
using Distances
using Econometrics
using EvoTrees 
using Gadfly
using GLM
using HypothesisTests
using Statistics
using Stella

popdisplay()
Gadfly.pop_theme()
Gadfly.push_theme(:dark)

#Data
file = "C:\\Users\\Dustin\\Documents\\GitHub\\Datasets\\CFB\\recruits.csv"
recruits = DataFrame(CSV.File(file, dateformat="yyyy/mm/dd"))

#SOME BASIC PLOTTING
#Plot (Height vs Weight)
set_default_plot_size(20cm,15cm)
plot_hw = Gadfly.plot(recruits, x=:Weight, y=:Height, color=recruits.Staff, Geom.point,
    Guide.title("Height vs Weight By Staff"), Guide.xlabel("Weight in lbs"), Guide.ylabel("Height in cm"))

#Plot (Height Distributions)
plot_hdist = Gadfly.plot(recruits, x=:Height, color=:Staff, Geom.density,
    Guide.title("Height Density By Staff"), Guide.xlabel("Height in cm"), Guide.ylabel("Density"))

#Plot (Weight Distributions)
plot_wdist = Gadfly.plot(recruits, x=:Weight, color=:Staff, Geom.density,
    Guide.title("Weight Density By Staff"), Guide.xlabel("Weight in lbs"), Guide.ylabel("Density"))

#Plot (Mean Height & Weight By Position & Staff)
recruitsGroupedPS = groupby(recruits, [:Pos,:Staff])
recruitsHW_means = combine([:Height, :Weight] => (h, w) -> (Height_mean=mean(h), Weight_mean=mean(w)), recruitsGroupedPS)

set_default_plot_size(45cm,12cm)
plot_pos_by_staff = Gadfly.plot(recruitsHW_means, xgroup="Pos", x="Weight_mean", y="Height_mean", color=:Staff,Geom.subplot_grid(Geom.point))

#Basic Statistics
#Check Min and Max
extrema(recruits.Height)
extrema(recruits.Weight)
extrema(recruits."Rating")
extrema(recruits."Stars")
extrema(recruits."RivalsRating")
extrema(recruits."RivalsStars")
extrema(recruits."ESPNStars")

#Check Mean
mean(recruits.Height)
mean(recruits.Weight)
mean(recruits."Rating")
mean(recruits."Stars")
mean(recruits."RivalsRating")
mean(recruits."RivalsStars")
mean(recruits."ESPNStars")

#Check Variance
var(recruits.Height)
var(recruits.Weight)
var(recruits."Rating")
var(recruits."Stars")
var(recruits."RivalsRating")
var(recruits."RivalsStars")
var(recruits."ESPNStars")

#Check Skewness
skewness(recruits.Height)
skewness(recruits.Weight)
skewness(recruits."Rating")
skewness(recruits."Stars")
skewness(recruits."RivalsRating")
skewness(recruits."RivalsStars")
skewness(recruits."ESPNStars")

#Check Tail Fatness
kurtosis(recruits.Height)
kurtosis(recruits.Weight)
kurtosis(recruits."Rating")
kurtosis(recruits."Stars")
kurtosis(recruits."RivalsRating")
kurtosis(recruits."RivalsStars")
kurtosis(recruits."ESPNStars")

#Check Entropy
entropy(recruits.Height)
entropy(recruits.Weight)
entropy(recruits."Rating")
entropy(recruits."Stars")
entropy(recruits."RivalsRating")
entropy(recruits."RivalsStars")
entropy(recruits."ESPNStars")

#Distribution Fitting
#Check if Normal
fit(Normal, recruits.Height)
fit(Normal, recruits.Weight)  
fit(Normal, recruits."Rating")
fit(Normal, recruits."Stars")
fit(Normal, recruits."RivalsRating")
fit(Normal, recruits."RivalsStars")
fit(Normal, recruits."ESPNRating")
fit(Normal, recruits."ESPNStars")

#Regression Models
#Simple Linear Regression: lm(@formula(), data)
lm(@formula(Weight ~ Height+Rating+Stars+RivalsRating+RivalsStars+ESPNRating+ESPNStars), recruits)
lm(@formula(Height ~ Weight+Rating+Stars+RivalsRating+RivalsStars+ESPNRating+ESPNStars), recruits)
lm(@formula(Rating ~ RivalsRating+ESPNRating), recruits)
lm(@formula(Stars ~ RivalsStars+ESPNStars), recruits)

#Multinominal Logistic Regression
recruitsCategorical = categorical(recruits)
fit(EconometricModel, @formula(Staff ~ Rating + Stars), recruitsCategorical, contrasts = Dict(:Staff => DummyCoding(base = "Oleary")))
fit(EconometricModel, @formula(Staff ~ RivalsRating + RivalsStars), recruitsCategorical, contrasts = Dict(:Staff => DummyCoding(base = "Oleary")))

#Generalized Linear Models by Staff
glm(@formula(Rating ~ Staff), recruits, Normal())
glm(@formula(RivalsRating ~ Staff), recruits, Normal())
glm(@formula(ESPNRating ~ Staff), recruits, Normal())

#Generalized Linear Models by State
glm(@formula(Rating ~ State), recruits, Normal())
glm(@formula(RivalsRating ~ State), recruits, Normal())
glm(@formula(ESPNRating ~ State), recruits, Normal())

#One-way Anova by Staff
Stella.anova(recruits, :Rating, :Staff)

#Kolmogorov–Smirnov Test of Normality
#Single Sample
ApproximateOneSampleKSTest(unique(recruits.Height), Normal())
ApproximateOneSampleKSTest(unique(recruits.Weight), Normal())
ApproximateOneSampleKSTest(unique(recruits.Rating), Normal())

#K-Means Clustering w/Plot
cluster_data = recruits[:,[4,5,11]]
features = collect(Matrix(cluster_data[:,1:3])')
results = kmeans(features, 3, maxiter=200, display=:iter)
Gadfly.plot(recruits, x="Weight",y ="Height", color=results.assignments, Geom.point)

a = results.assignments
b = results.centers
c = results.counts
d = pairwise(SqEuclidean(), features)

#Evaluate Silhouettes
mean(silhouettes(a, c, d))
#NOTE: Values fall in the [-1,1] and indicate how dense the clusters are. A score of 1 means the 
#      clusters are dense and well seperated. A score less than 0 means the data is poor. Try and
#      keep an eye on cluster scores below the mean, wide fluctuations and thickness of plot.

#Affinity Propagation
results2 = affinityprop(d; maxiter=200, tol=1e-6, damp=0.25, display=:iter)
Gadfly.plot(recruits,x="Weight",y ="Height", color=results2.assignments, Geom.point)

e = results2.exemplars
f = results2.assignments
g = results2.iterations
h = results2.converged

#Decision Tree Classification 
features = convert(Array, recruits[:, 11:16])
labels = convert(Array, recruits[:, 17])
model = DecisionTreeClassifier(max_depth=4)

DecisionTree.fit!(model, features, labels)
print_tree(model, 5)

learnedModel = [.85, 3.2, 4.0, 85, 4.0]
DecisionTree.predict(model, learnedModel)
predict_proba(model, learnedModel)
println(get_classes(model))
#NOTE: Applied a learned model based on prospective recruit infromation from 247Sports. The model
#      predicted that the prospective recruit would have been a Hupel prospect.
