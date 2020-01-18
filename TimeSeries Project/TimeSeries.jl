Boeing = readtimearray("BA.csv", format = "yyyy-mm-dd", delim = ',') #1962 - 2012
Caterpillar = readtimearray("CAT.csv", format = "yyyy-mm-dd", delim = ',') #1962 - 2012
Dell = readtimearray("DELL.csv", format = "yyyy-mm-dd", delim = ',') #1988 - 2015
Ebay = readtimearray("EBAY.csv", format = "yyyy-mm-dd", delim = ',') #1998 - 2016
Ford =readtimearray("F.csv", format = "yyyy-mm-dd", delim = ',') #1998 - 2016
GE = readtimearray("GE.csv", format = "yyyy-mm-dd", delim = ',') #1962 - 2016

#=
For this project we will be looking at the Boeing and Catipillar datasets
since they fall under the same timeframe.
=#

#Basic Plotting
plot(Boeing[:Open, :Close, :High, :Low], title = "Time Series (Boeing vs Caterpillar)",
        xaxis = "Timestamp", yaxis = "Price", palette = :dense)

plot!(Caterpillar[:Open, :Close, :High, :Low], palette = :turbid)

#Some Basic Methods
        #Variable order: Open   High    Low     Close
        #Autocovariance and Autocorrelation
        BA_close = readdlm("BA_close.txt")
        convert(Array{Real},BA_close)
        autocov_BA_close = autocov(BA_close, ; demean = true)
        autocor_BA_close = autocor(BA_close, ; demean = true)

        CAT_close = readdlm("CAT_close.txt")
        convert(Array{Real},CAT_close)
        autocov_CAT_close = autocov(CAT_close, ; demean = true)
        autocor_CAT_close = autocor(CAT_close, ; demean = true)

        plot(autocor_BA_close, title = "AutoCor of Historical Prices of Boeing Stock",
                xaxis = "lag", yaxis = "autocor", seriestype = :line)
        plot(autocov_BA_close, title = "AutoCov of Historical Prices of Boeing Stock",
                xaxis = "lag", yaxis = "autocov", seriestype = :line)

        #Cross-covariance and Cross-correlation
        crosscov_CAT_close = crosscov(CAT_close, BA_close, ; demean=true)
        plot(crosscov_CAT_close, title = "CrossCov of Historical Prices of Boeing vs Caterpillar",
                xaxis = "lag", yaxis = "autocov", seriestype = :line)
        crosscor_CAT_close = crosscor(CAT_close, BA_close, ;demean=true)
        plot(crosscor_CAT_close, title = "CrossCor of Historical Prices of Boeing vs Caterpillar",
                xaxis = "lag", yaxis = "autocor", seriestype = :line)

        #Partial Autocorrelation Function
        pacf_CAT_close_reg = pacf(CAT_close,[2,3,4,5,10,15,20,25,30]; method=:regression)
        plot(pacf_CAT_close_reg, title = "PACF_Regression of Caterpillar",
                xaxis = "lag", yaxis = "pacf", seriestype = :bar)
        pacf_CAT_close_yule = pacf(CAT_close,[2,3,4,5,10,15,20,25,30]; method=:yulewalker)
        plot(pacf_CAT_close_yule, title = "PACF_Yule of Caterpillar",
                xaxis = "lag", yaxis = "pacf", seriestype = :bar)
        pacf_BA_close_reg = pacf(BA_close,[2,3,4,5,10,15,20,25,30]; method=:regression)
        plot(pacf_BA_close_reg, title = "PACF_Regression of Boeing",
                xaxis = "lag", yaxis = "pacf", seriestype = :bar)
        pacf_BA_close_yule = pacf(BA_close,[2,3,4,5,10,15,20,25,30]; method=:yulewalker)
        plot(pacf_BA_close_yule, title = "PACF_Yule of Boeing",
                xaxis = "lag", yaxis = "pacf", seriestype = :bar)

        #ARMA Model Example for Boeing Data
        phi = 0.5
        theta = [0.0, -0.5]
        sigma = 1.0
        model1 = ARMA(phi, theta, sigma)
        quad_plot(model1)

        #ARCH and GARCH Models of Boeing Data
        BA_close2 = readdlm("BA_close.txt")[:,1]
        CAT_close2 = readdlm("CAT_close.txt")[:,1]
        autocor(BA_close.^2, 0:10, demean=true)
        autocor(CAT_close.^2, 0:10, demean=true)

        ARCHLMTest(BA_close2, 1)
        #The null is strongly rejected, again providing evidence for the presence of volatility clustering at lag=1
        ARCHLMTest(CAT_close2, 1)
        #The null is strongly rejected, again providing evidence for the presence of volatility clustering at lag=1

        fit(GARCH{1, 1}, BA_close2)
        fit(GARCH{1, 1}, CAT_close2)

        #Regression
        X = ones(length(BA_close2), 1)
        reg = Regression(X)
        fit(GARCH{1, 1}, BA_close2; meanspec=reg)
        fit(GARCH{1, 1}, CAT_close2; meanspec=reg)
        #Both the CAT and BA datasets have the same number of observations on the same timeframe.
        #Notice that because in this case X contains only a column of ones, the estimation results,
        #are equivalent to those obtained with our fits above.
        fit(EGARCH{1, 1, 1}, BA_close2; meanspec=NoIntercept, dist=StdT)
        fit(EGARCH{1, 1, 1}, CAT_close2; meanspec=NoIntercept, dist=StdT)

        selectmodel(ARCH, BA_close2; criterion=aic, maxlags=2, dist=StdT)
        selectmodel(GARCH, BA_close2; criterion=aic, maxlags=2, dist=StdT)
        selectmodel(EGARCH, BA_close2; criterion=aic, maxlags=2, dist=StdT)
        selectmodel(ARCH, CAT_close2; criterion=aic, maxlags=2, dist=StdT)
        selectmodel(GARCH, CAT_close2; criterion=aic, maxlags=2, dist=StdT)
        selectmodel(EGARCH, CAT_close2; criterion=aic, maxlags=2, dist=StdT)


#Functions for ARMA building
function plot_spectral_density(arma::ARMA)
    (w, spect) = spectral_density(arma, two_pi=false)
    p = plot(w, spect, color=:blue, linewidth=2, alpha=0.7,
             xlims=(0, pi), xlabel="frequency", ylabel="spectrum",
             title="Spectral density", yscale=:log, legend=:none, grid=false)
    return p
end

function plot_autocovariance(arma::ARMA)
    acov = autocovariance(arma)
    n = length(acov)
    N = repeat(0:(n - 1), 1, 2)'
    heights = [zeros(1,n); acov']
    p = scatter(0:(n - 1), acov, title="Autocovariance", xlims=(-0.5, n - 0.5),
                xlabel="time", ylabel="autocovariance", legend=:none, color=:blue)
    plot!(-1:(n + 1), zeros(1, n + 3), color=:red, linewidth=0.5)
    plot!(N, heights, color=:blue, grid=false)
    return p
end

function plot_impulse_response(arma::ARMA)
    psi = impulse_response(arma)
    n = length(psi)
    N = repeat(0:(n - 1), 1, 2)'
    heights = [zeros(1,n); psi']
    p = scatter(0:(n - 1), psi, title="Impulse response", xlims=(-0.5, n - 0.5),
                xlabel="time", ylabel="response", legend=:none, color=:blue)
    plot!(-1:(n + 1), zeros(1, n + 3), color=:red, linewidth=0.5)
    plot!(N, heights, color=:blue, grid=false)
    return p
end

function plot_simulation(arma::ARMA)
    X = simulation(arma)
    n = length(X)
    p = plot(0:(n - 1), X, color=:blue, linewidth=2, alpha=0.7,
             xlims=(0.0, n), xlabel="time", ylabel="state space",
            title="Sample path", legend=:none, grid=:false)
    return p
end

function quad_plot(arma::ARMA)
    p = plot(plot_impulse_response(arma), plot_autocovariance(arma),
         plot_spectral_density(arma), plot_simulation(arma))
    return p
end

