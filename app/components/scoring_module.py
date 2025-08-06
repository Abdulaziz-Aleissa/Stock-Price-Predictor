"""
Scoring Module
Handle stock scoring calculations only
"""

from app.utils.stock_scoring import StockScoring
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class StockScoringManager:
    """Handle stock scoring calculations"""
    
    def __init__(self):
        self.scorer = StockScoring()
    
    def analyze_single_stock(self, symbol: str) -> Dict:
        """Analyze a single stock and return scoring results"""
        try:
            results = self.scorer.analyze_stocks([symbol])
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            return None
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple stocks and return scoring results"""
        try:
            # Limit to reasonable number of stocks
            if len(symbols) > 10:
                logger.warning(f"Too many symbols requested: {len(symbols)}, limiting to 10")
                symbols = symbols[:10]
            
            return self.scorer.analyze_stocks(symbols)
        except Exception as e:
            logger.error(f"Error analyzing stocks {symbols}: {str(e)}")
            return []
    
    def get_scoring_criteria(self) -> Dict:
        """Get the scoring criteria used by the scorer"""
        try:
            return self.scorer.scoring_criteria
        except Exception as e:
            logger.error(f"Error getting scoring criteria: {str(e)}")
            return {}
    
    def compare_stocks(self, symbols: List[str]) -> Dict:
        """Compare multiple stocks and provide ranking"""
        try:
            results = self.analyze_multiple_stocks(symbols)
            
            if not results:
                return {'error': 'No valid analysis results'}
            
            # Sort by total score descending
            sorted_results = sorted(results, key=lambda x: x.get('total_score', 0), reverse=True)
            
            comparison = {
                'ranking': sorted_results,
                'best_stock': sorted_results[0] if sorted_results else None,
                'worst_stock': sorted_results[-1] if sorted_results else None,
                'analysis_summary': {
                    'total_analyzed': len(sorted_results),
                    'average_score': sum([r.get('total_score', 0) for r in sorted_results]) / len(sorted_results) if sorted_results else 0,
                    'score_range': {
                        'highest': sorted_results[0].get('total_score', 0) if sorted_results else 0,
                        'lowest': sorted_results[-1].get('total_score', 0) if sorted_results else 0
                    }
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing stocks {symbols}: {str(e)}")
            return {'error': f'Comparison failed: {str(e)}'}
    
    def get_top_scored_stocks(self, symbols: List[str], top_n: int = 5) -> List[Dict]:
        """Get top N scored stocks from the given list"""
        try:
            results = self.analyze_multiple_stocks(symbols)
            
            if not results:
                return []
            
            # Sort by total score descending and return top N
            sorted_results = sorted(results, key=lambda x: x.get('total_score', 0), reverse=True)
            return sorted_results[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top scored stocks: {str(e)}")
            return []
    
    def get_stocks_by_score_range(self, symbols: List[str], min_score: float, max_score: float) -> List[Dict]:
        """Get stocks within a specific score range"""
        try:
            results = self.analyze_multiple_stocks(symbols)
            
            if not results:
                return []
            
            filtered_results = [
                result for result in results 
                if min_score <= result.get('total_score', 0) <= max_score
            ]
            
            return sorted(filtered_results, key=lambda x: x.get('total_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error filtering stocks by score range: {str(e)}")
            return []
    
    def validate_symbols(self, symbols: List[str]) -> Dict:
        """Validate stock symbols before analysis"""
        try:
            valid_symbols = []
            invalid_symbols = []
            
            for symbol in symbols:
                symbol = symbol.strip().upper()
                if symbol:
                    # Basic validation - could be enhanced
                    if len(symbol) <= 5 and symbol.isalpha():
                        valid_symbols.append(symbol)
                    else:
                        invalid_symbols.append(symbol)
            
            return {
                'valid_symbols': valid_symbols,
                'invalid_symbols': invalid_symbols,
                'validation_passed': len(invalid_symbols) == 0
            }
            
        except Exception as e:
            logger.error(f"Error validating symbols: {str(e)}")
            return {
                'valid_symbols': [],
                'invalid_symbols': symbols,
                'validation_passed': False
            }


# Global instance to be used across the application
stock_scoring_manager = StockScoringManager()