model=${1:-gpt-4o-mini}
file=${3:-test.csv}

echo "🤖 Model to test: $model"
# echo "🤖 Generating SOAP..."
# python3 soap.py --model "$model" --file "$file"

echo "🤖 Judge: $judge"
echo "🚀 Starting SOAP Error Type Analysis..."

echo "🧪 Splitting into Units..."
python3 unit.py --model "$judge" --file "$model""_""$file"

echo "🔍 Analyzing Error Types..." 
python3 eval.py --model "$judge" --file "$judge""_""$model""_""$file"

echo "📊 Scoring..."
python3 score.py --file "$judge""_""$model""_""$file"

echo "✅ SOAP Error Type Analysis completed successfully!"