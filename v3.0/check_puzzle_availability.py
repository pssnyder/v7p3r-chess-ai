"""
V7P3R Puzzle Availability Checker
=================================

Check how many puzzles are available before starting training
"""

import sys
from pathlib import Path

# Add paths for database access
sys.path.insert(0, str(Path(__file__).parent / "v3.0" / "src"))

def check_puzzle_availability(target_themes=None, excluded_themes=None, min_rating=None, max_rating=None):
    """Check how many puzzles match the training criteria"""
    
    try:
        from database.enhanced_puzzle_db_v2 import EnhancedPuzzleDatabaseV2
        
        # Connect to database
        db_path = "v3.0/data/v7p3rai_puzzle_training_v2.db"
        if not Path(db_path).exists():
            print(f"❌ Database not found: {db_path}")
            return 0
            
        db = EnhancedPuzzleDatabaseV2(db_path)
        
        # Build query conditions
        conditions = []
        params = []
        
        if min_rating:
            conditions.append("rating >= ?")
            params.append(min_rating)
            
        if max_rating:
            conditions.append("rating <= ?")
            params.append(max_rating)
            
        if target_themes:
            theme_conditions = []
            for theme in target_themes:
                theme_conditions.append("themes LIKE ?")
                params.append(f"%{theme}%")
            conditions.append(f"({' OR '.join(theme_conditions)})")
            
        if excluded_themes:
            for theme in excluded_themes:
                conditions.append("themes NOT LIKE ?")
                params.append(f"%{theme}%")
        
        # Add condition to exclude already attempted puzzles
        conditions.append("puzzle_id NOT IN (SELECT DISTINCT puzzle_id FROM training_history)")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Count total matching puzzles
        query = f"""
        SELECT COUNT(*) as total_count,
               MIN(rating) as min_rating,
               MAX(rating) as max_rating,
               AVG(rating) as avg_rating
        FROM puzzles 
        WHERE {where_clause}
        """
        
        result = db.execute_query(query, params)
        
        if result:
            row = result[0]
            total_count = row[0]
            min_r = row[1] if row[1] else "N/A"
            max_r = row[2] if row[2] else "N/A"
            avg_r = row[3] if row[3] else "N/A"
            
            print(f"📊 PUZZLE AVAILABILITY REPORT")
            print(f"=" * 50)
            print(f"🧩 Total available puzzles: {total_count:,}")
            print(f"📈 Rating range: {min_r} - {max_r}")
            print(f"📊 Average rating: {avg_r:.1f}" if isinstance(avg_r, (int, float)) else f"📊 Average rating: {avg_r}")
            
            # Get theme breakdown
            if total_count > 0:
                theme_query = f"""
                SELECT themes, COUNT(*) as count
                FROM puzzles 
                WHERE {where_clause}
                GROUP BY themes
                ORDER BY count DESC
                LIMIT 10
                """
                
                theme_results = db.execute_query(theme_query, params)
                
                if theme_results:
                    print(f"\n🎨 TOP THEME COMBINATIONS:")
                    print(f"-" * 30)
                    for theme_row in theme_results:
                        themes = theme_row[0]
                        count = theme_row[1]
                        # Truncate long theme strings
                        display_themes = themes[:50] + "..." if len(themes) > 50 else themes
                        print(f"  {count:,} puzzles: {display_themes}")
            
            return total_count
            
        else:
            print("❌ Error querying database")
            return 0
            
    except Exception as e:
        print(f"❌ Error checking puzzle availability: {e}")
        return 0
    finally:
        try:
            db.close()
        except:
            pass

def get_training_recommendations(available_count, target_themes, max_rating):
    """Provide training recommendations based on availability"""
    
    print(f"\n💡 TRAINING RECOMMENDATIONS")
    print(f"=" * 40)
    
    if available_count == 0:
        print("🔴 CRITICAL: No puzzles available!")
        print("Recommendations:")
        print("  • Remove theme restrictions")
        print("  • Increase max rating")
        print("  • Clear training history to reuse puzzles")
        return "STOP"
        
    elif available_count < 100:
        print("🟡 WARNING: Very few puzzles available!")
        print(f"  Only {available_count} puzzles match criteria")
        print("Recommendations:")
        print("  • Reduce training time (--hpts 1)")
        print("  • Use smaller batches (--batch-size 10)")
        print("  • Consider broader criteria")
        return "CAUTION"
        
    elif available_count < 1000:
        print("🟠 NOTICE: Limited puzzle set")
        print(f"  {available_count} puzzles available")
        print("Recommendations:")
        print("  • Moderate training time (--hpts 2-3)")
        print("  • Standard batches (--batch-size 20-30)")
        return "LIMITED"
        
    else:
        print("✅ GOOD: Adequate puzzles available!")
        print(f"  {available_count:,} puzzles available")
        print("Recommendations:")
        print("  • Full training session possible")
        print("  • Normal batch sizes recommended")
        return "GOOD"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check puzzle availability before training")
    parser.add_argument('--target-themes', type=str, help='Comma-separated themes to check')
    parser.add_argument('--excluded-themes', type=str, help='Comma-separated themes to exclude')
    parser.add_argument('--max-rating', type=int, help='Maximum puzzle rating')
    parser.add_argument('--min-rating', type=int, help='Minimum puzzle rating')
    
    args = parser.parse_args()
    
    # Parse themes
    target_themes = args.target_themes.split(',') if args.target_themes else None
    excluded_themes = args.excluded_themes.split(',') if args.excluded_themes else None
    
    print("🔍 V7P3R PUZZLE AVAILABILITY CHECK")
    print("=" * 50)
    print(f"Target themes: {target_themes}")
    print(f"Excluded themes: {excluded_themes}")
    print(f"Rating range: {args.min_rating or 'No min'} - {args.max_rating or 'No max'}")
    print()
    
    # Check availability
    available_count = check_puzzle_availability(
        target_themes=target_themes,
        excluded_themes=excluded_themes,
        min_rating=args.min_rating,
        max_rating=args.max_rating
    )
    
    # Get recommendations
    status = get_training_recommendations(available_count, target_themes, args.max_rating)
    
    print(f"\n🎯 STATUS: {status}")
    
    if status == "STOP":
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit(main())