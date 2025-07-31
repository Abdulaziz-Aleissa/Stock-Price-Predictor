"""
Comprehensive Stock Risk Analysis
Implements two main risk analysis strategies:
1. Fundamental Risk Analysis (Altman Z-Score + Interest Coverage)
2. Technical Risk Analysis (ATR + Support Zone)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Comprehensive risk analysis for stocks using fundamental and technical indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fundamental_risk_analysis(self, symbol, working_capital=None, retained_earnings=None, 
                                ebit=None, market_cap=None, total_assets=None, 
                                total_liabilities=None, sales=None, interest_expense=None):
        """
        Perform fundamental risk analysis using Altman Z-Score and Interest Coverage Ratio
        
        Args:
            symbol: Stock ticker symbol
            working_capital: Working Capital in millions
            retained_earnings: Retained Earnings in millions
            ebit: Earnings Before Interest and Tax in millions
            market_cap: Market Value of Equity in millions
            total_assets: Total Assets in millions
            total_liabilities: Total Liabilities in millions
            sales: Sales/Revenue in millions
            interest_expense: Interest Expense in millions
            
        Returns:
            Dictionary with fundamental risk analysis results
        """
        try:
            # Validate required inputs
            required_fields = {
                'Working Capital': working_capital,
                'Retained Earnings': retained_earnings,
                'EBIT': ebit,
                'Market Cap': market_cap,
                'Total Assets': total_assets,
                'Total Liabilities': total_liabilities,
                'Sales': sales,
                'Interest Expense': interest_expense
            }
            
            missing_fields = [field for field, value in required_fields.items() if value is None]
            if missing_fields:
                return {"error": f"Missing required fields: {', '.join(missing_fields)}"}
            
            # Convert to float and validate positive values where appropriate
            try:
                wc = float(working_capital)
                re = float(retained_earnings)
                ebit_val = float(ebit)
                mve = float(market_cap)
                ta = float(total_assets)
                tl = float(total_liabilities)
                sales_val = float(sales)
                ie = float(interest_expense)
            except ValueError:
                return {"error": "All inputs must be valid numbers"}
            
            if ta <= 0:
                return {"error": "Total Assets must be positive"}
            if tl <= 0:
                return {"error": "Total Liabilities must be positive"}
            if ie <= 0:
                return {"error": "Interest Expense must be positive for Interest Coverage calculation"}
            
            # Calculate Altman Z-Score
            # Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(S/TA)
            
            wc_ta = wc / ta
            re_ta = re / ta
            ebit_ta = ebit_val / ta
            mve_tl = mve / tl
            s_ta = sales_val / ta
            
            z_score = (1.2 * wc_ta + 
                      1.4 * re_ta + 
                      3.3 * ebit_ta + 
                      0.6 * mve_tl + 
                      1.0 * s_ta)
            
            # Interpret Z-Score
            if z_score < 1.8:
                z_interpretation = "Distress Zone - High bankruptcy risk"
                z_risk_level = "High"
            elif z_score <= 3.0:
                z_interpretation = "Caution Zone - Moderate financial risk"
                z_risk_level = "Medium"
            else:
                z_interpretation = "Safe Zone - Low financial risk"
                z_risk_level = "Low"
            
            # Calculate Interest Coverage Ratio
            interest_coverage = ebit_val / ie
            
            # Interpret Interest Coverage Ratio
            if interest_coverage < 1.5:
                ic_interpretation = "High Financial Risk - Difficulty covering interest payments"
                ic_risk_level = "High"
            elif interest_coverage <= 5.0:
                ic_interpretation = "Moderate Risk - Adequate interest coverage"
                ic_risk_level = "Medium"
            else:
                ic_interpretation = "Low Financial Risk - Strong interest coverage"
                ic_risk_level = "Low"
            
            # Overall fundamental risk assessment
            risk_scores = {"High": 3, "Medium": 2, "Low": 1}
            overall_score = (risk_scores[z_risk_level] + risk_scores[ic_risk_level]) / 2
            
            if overall_score >= 2.5:
                overall_risk = "High"
                overall_interpretation = "⚠️ High fundamental risk - Exercise caution"
            elif overall_score >= 1.5:
                overall_risk = "Medium"
                overall_interpretation = "⚡ Moderate fundamental risk - Monitor closely"
            else:
                overall_risk = "Low"
                overall_interpretation = "✅ Low fundamental risk - Financially stable"
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "analysis_type": "FUNDAMENTAL RISK ANALYSIS",
                "altman_z_score": {
                    "value": round(z_score, 3),
                    "components": {
                        "working_capital_to_assets": round(wc_ta, 4),
                        "retained_earnings_to_assets": round(re_ta, 4),
                        "ebit_to_assets": round(ebit_ta, 4),
                        "market_value_to_liabilities": round(mve_tl, 4),
                        "sales_to_assets": round(s_ta, 4)
                    },
                    "interpretation": z_interpretation,
                    "risk_level": z_risk_level
                },
                "interest_coverage_ratio": {
                    "value": round(interest_coverage, 2),
                    "calculation": f"{ebit_val:.1f}M / {ie:.1f}M",
                    "interpretation": ic_interpretation,
                    "risk_level": ic_risk_level
                },
                "overall_assessment": {
                    "risk_level": overall_risk,
                    "interpretation": overall_interpretation,
                    "score": round(overall_score, 2)
                },
                "inputs_summary": {
                    "working_capital": f"${wc:.1f}M",
                    "retained_earnings": f"${re:.1f}M",
                    "ebit": f"${ebit_val:.1f}M",
                    "market_cap": f"${mve:.1f}M",
                    "total_assets": f"${ta:.1f}M",
                    "total_liabilities": f"${tl:.1f}M",
                    "sales": f"${sales_val:.1f}M",
                    "interest_expense": f"${ie:.1f}M"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental risk analysis error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def technical_risk_analysis(self, symbol, support_level=None):
        """
        Perform technical risk analysis using ATR and Support Zone analysis
        
        Args:
            symbol: Stock ticker symbol
            support_level: Manual input for recent strong support level
            
        Returns:
            Dictionary with technical risk analysis results
        """
        try:
            # Get 6 months of daily stock data
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # ~6 months
            
            try:
                data = stock.history(start=start_date, end=end_date)
            except Exception as e:
                # Handle network connectivity issues gracefully
                self.logger.warning(f"Network error fetching data for {symbol}: {str(e)}")
                return {"error": f"Unable to fetch market data for {symbol}. Please check your internet connection or try again later."}
            
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            if len(data) < 14:
                return {"error": f"Insufficient data for {symbol} - need at least 14 days"}
            
            # Calculate 14-day ATR (Average True Range)
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # True Range calculation
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr_14 = true_range.rolling(window=14).mean()
            
            # Get current price (last close)
            current_price = float(close.iloc[-1])
            current_atr = float(atr_14.iloc[-1])
            
            # Handle support level input
            if support_level is None:
                # If no support level provided, use 6-month low as a rough estimate
                support_level = float(low.min())
                support_level_source = "6-month low (estimated)"
            else:
                try:
                    support_level = float(support_level)
                    support_level_source = "User provided"
                except ValueError:
                    return {"error": "Support level must be a valid number"}
            
            if support_level <= 0:
                return {"error": "Support level must be positive"}
            
            if support_level >= current_price:
                return {"error": "Support level must be below current price"}
            
            # Calculate risk metrics
            downside_risk_dollars = current_price - support_level
            atr_risk_units = downside_risk_dollars / current_atr
            risk_days = atr_risk_units  # Simplified: assumes 1 ATR move per day on average
            
            # Risk interpretation based on ATR units
            if atr_risk_units > 3:
                risk_interpretation = "⚠️ High volatility risk - Price could fall quickly to support"
                risk_level = "High"
            elif atr_risk_units > 2:
                risk_interpretation = "⚡ Moderate volatility risk - Monitor price action closely"
                risk_level = "Medium"
            else:
                risk_interpretation = "✅ Low volatility risk - Support level relatively close"
                risk_level = "Low"
            
            # Additional ATR-based insights
            atr_percentage = (current_atr / current_price) * 100
            
            if atr_percentage > 4:
                volatility_assessment = "Very High - Expect large daily price swings"
            elif atr_percentage > 2.5:
                volatility_assessment = "High - Significant daily volatility"
            elif atr_percentage > 1.5:
                volatility_assessment = "Moderate - Normal volatility levels"
            else:
                volatility_assessment = "Low - Relatively stable price action"
            
            # Calculate historical volatility metrics
            returns = close.pct_change().dropna()
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252) * 100
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "analysis_type": "TECHNICAL RISK ANALYSIS",
                "current_price": round(current_price, 2),
                "support_level": {
                    "value": round(support_level, 2),
                    "source": support_level_source
                },
                "atr_analysis": {
                    "atr_14_day": round(current_atr, 2),
                    "atr_percentage": round(atr_percentage, 2),
                    "volatility_assessment": volatility_assessment
                },
                "risk_metrics": {
                    "downside_risk_dollars": round(downside_risk_dollars, 2),
                    "downside_risk_percentage": round((downside_risk_dollars / current_price) * 100, 2),
                    "atr_risk_units": round(atr_risk_units, 2),
                    "estimated_risk_days": round(risk_days, 1),
                    "risk_level": risk_level,
                    "interpretation": risk_interpretation
                },
                "volatility_metrics": {
                    "daily_volatility": round(daily_volatility * 100, 2),
                    "annualized_volatility": round(annualized_volatility, 1)
                },
                "analysis_period": {
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "trading_days": len(data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Technical risk analysis error: {str(e)}")
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def comprehensive_risk_analysis(self, symbol, **kwargs):
        """
        Perform both fundamental and technical risk analysis
        
        Args:
            symbol: Stock ticker symbol
            **kwargs: All parameters for both fundamental and technical analysis
            
        Returns:
            Dictionary with both analyses
        """
        try:
            results = {
                "symbol": symbol.upper(),
                "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "fundamental_analysis": None,
                "technical_analysis": None
            }
            
            # Extract fundamental analysis parameters
            fundamental_params = {}
            fundamental_fields = ['working_capital', 'retained_earnings', 'ebit', 'market_cap', 
                                'total_assets', 'total_liabilities', 'sales', 'interest_expense']
            
            for field in fundamental_fields:
                if field in kwargs and kwargs[field] is not None:
                    fundamental_params[field] = kwargs[field]
            
            # Perform fundamental analysis if we have the required data
            if len(fundamental_params) == len(fundamental_fields):
                results["fundamental_analysis"] = self.fundamental_risk_analysis(symbol, **fundamental_params)
            
            # Extract technical analysis parameters
            support_level = kwargs.get('support_level')
            
            # Always perform technical analysis (support level is optional)
            results["technical_analysis"] = self.technical_risk_analysis(symbol, support_level)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive risk analysis error: {str(e)}")
            return {"error": f"Comprehensive analysis failed: {str(e)}"}


# Global instance
risk_analyzer = RiskAnalyzer()