for dir in */; do 
    echo "📂 Processing directory: $dir"
    (cd "$dir" && echo "  🚀 Running deploy in $(pwd)" && infsh deploy && echo "  ✅ Finished $dir") || echo "  ❌ Failed in $dir"
done