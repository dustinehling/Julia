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
using Gadfly
using GLM
using HypothesisTests
using Statistics

popdisplay()
Gadfly.pop_theme()
Gadfly.push_theme(:dark)

#Data
file = "C:\\Users\\dusti\\Documents\\github\\Datasets\\CFB\\recruits.csv"
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
recruitsHW_means = combine(recruitsGroupedPS, [:Height, :Weight] .=> mean)

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

#One-way Anova by Staff
scoresOleary = AbstractVector{Real}(recruits[recruits[!,:Staff] .== "Oleary",:].Rating)
scoresFrost = AbstractVector{Real}(recruits[recruits[!,:Staff] .== "Frost",:].Rating)
scoresHuepel = AbstractVector{Real}(recruits[recruits[!,:Staff] .== "Huepel",:].Rating)
scoresMalzahn = AbstractVector{Real}(recruits[recruits[!,:Staff] .== "Malzahn",:].Rating)
OneWayANOVATest(scoresOleary, scoresFrost, scoresHuepel, scoresMalzahn)
#Note: In general, one-way anova is only robust against violations of equal variance assumption when 
#       sample size is equivilent across levels. Thus more analysis must be done to determine if
#       unequal variances exist between levels. In the event they do we must look to the non-parametric 
#       Kruskal-Wallis Test of differences between means between the 4 groups.

#Test for variance equivilence
LeveneTest(scoresOleary, scoresFrost, scoresHuepel, scoresMalzahn)
BrownForsytheTest(scoresOleary, scoresFrost, scoresHuepel, scoresMalzahn)
#Note: We can see from the output generated that the use of distibution mean as the paramter of differientiation
#       infers that the variance of each rating distribution is not equal. However, when the median parameter is used 
#       we fail to infer this alternative hypothesis. The optimal choice in equal variance testing depends on the underlying 
#       distribution but it is general practice that the Brown-Forsythe Test is recommended since it is the more robust
#       choice and maintains the higher degree of statistical power.

#Test for variance equivilence (cont.)
UnequalVarianceTTest(scoresOleary, scoresFrost)
UnequalVarianceTTest(scoresOleary, scoresHuepel)
UnequalVarianceTTest(scoresOleary, scoresMalzahn)

UnequalVarianceTTest(scoresFrost, scoresHuepel)
UnequalVarianceTTest(scoresFrost, scoresMalzahn)

UnequalVarianceTTest(scoresHuepel, scoresMalzahn)
#Note: For groups with differing sample sizes, another test of interest is Welch's t-test, an adaption of the Student's t-test 
#       that is more reeliable in the presence of unequal variances. This test assumes however, that the underlying sample disributions
#       are normal. From the testing, we infer that the variance among the samples are unequal.
#       
#       Conclusion: Levene's Test > Brown-Forsythe Test
#
#       - Variance among Oleary & Frost recruits can be assumed to be equal, while Hupuel & Malzahn recruits reject this null assumption.
#       - Oddly, Huepel variance is assumed to be equal to Frost variance given the above
#                       

#Kolmogorov–Smirnov Test of Normality
#Single Sample
ApproximateOneSampleKSTest(unique(recruits.Rating), Normal())

#Two Sample
ApproximateTwoSampleKSTest(scoresOleary, scoresFrost)
ApproximateTwoSampleKSTest(scoresOleary, scoresHuepel)
ApproximateTwoSampleKSTest(scoresOleary, scoresMalzahn)

ApproximateTwoSampleKSTest(scoresFrost, scoresHuepel)
ApproximateTwoSampleKSTest(scoresFrost, scoresMalzahn)

ApproximateTwoSampleKSTest(scoresHuepel, scoresMalzahn)
#Note: Single sample, commonly called the goodness of fit test, infers if the distribution differes substantially from the theoretical expectations of a Gaussian distribution.
#       Two sample, compares two groups against eachother to determine if the samples were taken from the same underlying population with the same disribution. If the p-value is   
#       small we conclude that the samples are from distinct populations that may differ in median, shape or variance.
#
#       - Reject null that the rating values come from a normal disribution. Subsets for each coach also are not normal.
#
#       - Frost recruits show no statistical deviation from Oleary recruits
#       - Huepel & Malzahn recruits come from a distinct distributions relative to Oleary & Frost recruits

#Test for mean equivilence
KruskalWallisTest(scoresOleary, scoresHuepel, scoresHuepel, scoresMalzahn)
#Assumptions:
#       - Samples were randomly taken
#       - Samples are mutually independent 
#       - Scale is ordinal and the variable is continuous 
#       - Note: If the test is used as a test of dominance, it has no distributional assumptions. If it used to compare medians, 
#           the distributions must be similar apart from their locations.
#       - Note: The test is generally considered to be robust to ties. However, if ties are present they should not be concentrated 
#           together in one part of the distribution (normal or uniform).
#
#       Conclusion: We reject the null hypothesis that the groups originate from the same distribution, and in turn conclude that one
#           or more of the sub-strata have a different mean/median value.

#Regression Models
#Simple Linear Regression: lm(@formula(), data)
lm(@formula(Weight ~ Height+Rating+Stars+RivalsRating+RivalsStars+ESPNRating+ESPNStars), recruits)
lm(@formula(Height ~ Weight+Rating+Stars+RivalsRating+RivalsStars+ESPNRating+ESPNStars), recruits)
lm(@formula(Rating ~ RivalsRating+ESPNRating), recruits)
lm(@formula(Stars ~ RivalsStars+ESPNStars), recruits)

#Multinominal Logistic Regression
recruitsCategorical = transform!(recruits, :Staff => categorical, renamecols=false)
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
#      predicted that the prospective recruit would have been a Heupel prospect.