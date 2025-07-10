#!/bin/bash

# =============================================================================
# Chunk Number Update Script
# =============================================================================
# 功能: 批量更新作业脚本中的chunk编号
# 使用方法: ./update_chunk.sh <旧chunk号> <新chunk号> [--dry-run]
# 使用方法: ./update_chunk.sh <旧chunk号> <新chunk号>
# 例如: ./update_chunk.sh 00 01 --dry-run
# =============================================================================

set -e  # 遇到错误时立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    print_error "参数错误！"
    echo "使用方法: $0 <旧chunk号> <新chunk号> [--dry-run]"
    echo ""
    echo "示例:"
    echo "  $0 00 01           # 将chunk_00替换为chunk_01"
    echo "  $0 01 02           # 将chunk_01替换为chunk_02"
    echo "  $0 1 2             # 自动补零，将chunk_01替换为chunk_02"
    echo "  $0 00 01 --dry-run # 预览模式，只显示会替换什么，不实际执行"
    exit 1
fi

OLD_CHUNK=$1
NEW_CHUNK=$2
DRY_RUN=false

# 检查是否是dry-run模式
if [ $# -eq 3 ] && [ "$3" = "--dry-run" ]; then
    DRY_RUN=true
fi

# 自动补零，确保是两位数格式
if [ ${#OLD_CHUNK} -eq 1 ]; then
    OLD_CHUNK="0${OLD_CHUNK}"
fi

if [ ${#NEW_CHUNK} -eq 1 ]; then
    NEW_CHUNK="0${NEW_CHUNK}"
fi

# 验证chunk格式（应该是两位数字）
if ! [[ "$OLD_CHUNK" =~ ^[0-9]{2}$ ]] || ! [[ "$NEW_CHUNK" =~ ^[0-9]{2}$ ]]; then
    print_error "chunk号格式错误！应该是两位数字，如：00, 01, 02..."
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    print_info "预览模式：chunk_${OLD_CHUNK} → chunk_${NEW_CHUNK} (不会实际修改文件)"
else
    print_info "开始更新：chunk_${OLD_CHUNK} → chunk_${NEW_CHUNK}"
fi

# 定义需要更新的文件列表
declare -a FILES_TO_UPDATE=(
    "leo.sh"
    "scripts/gen_vllm_ray_visual.sh"
)

# 定义需要替换的模式
declare -a PATTERNS=(
    "cc12m_${OLD_CHUNK}"
    "chunk_${OLD_CHUNK}"
    "gen_cc12m_${OLD_CHUNK}"
)

# 备份标志
BACKUP_CREATED=false
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

# 创建备份函数
create_backup() {
    if [ "$BACKUP_CREATED" = false ]; then
        mkdir -p "$BACKUP_DIR"
        print_info "创建备份目录: $BACKUP_DIR"
        BACKUP_CREATED=true
    fi
}

# 更新文件函数
update_file() {
    local file=$1
    local changes_made=false
    
    if [ ! -f "$file" ]; then
        print_warning "文件不存在: $file"
        return
    fi
    
    # 检查文件是否包含旧的chunk号
    local contains_old_chunk=false
    for pattern in "${PATTERNS[@]}"; do
        if grep -q "$pattern" "$file"; then
            contains_old_chunk=true
            break
        fi
    done
    
    if [ "$contains_old_chunk" = false ]; then
        print_warning "$file 中未找到 chunk_${OLD_CHUNK} 相关内容"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        # 预览模式：只显示会替换什么
        print_info "预览 $file 中的替换:"
        for pattern in "${PATTERNS[@]}"; do
            local new_pattern="${pattern/_${OLD_CHUNK}/_${NEW_CHUNK}}"
            if grep -q "$pattern" "$file"; then
                echo "  📝 会替换: $pattern → $new_pattern"
                # 显示匹配的行
                grep -n "$pattern" "$file" | while read -r line; do
                    echo "    ${line}"
                done
                changes_made=true
            fi
        done
    else
        # 实际执行模式：创建备份并替换
        create_backup
        cp "$file" "$BACKUP_DIR/"
        print_info "已备份: $file → $BACKUP_DIR/"
        
        # 显示将要进行的替换
        print_info "正在更新 $file ..."
        
        # 执行替换
        for pattern in "${PATTERNS[@]}"; do
            local new_pattern="${pattern/_${OLD_CHUNK}/_${NEW_CHUNK}}"
            if grep -q "$pattern" "$file"; then
                sed -i "s/${pattern}/${new_pattern}/g" "$file"
                print_info "  ✓ $pattern → $new_pattern"
                changes_made=true
            fi
        done
    fi
    
    if [ "$changes_made" = true ]; then
        if [ "$DRY_RUN" = true ]; then
            print_info "预览完成: $file"
        else
            print_success "已更新: $file"
        fi
    fi
}

# 主要更新流程
print_info "开始检查和更新文件..."
echo ""

for file in "${FILES_TO_UPDATE[@]}"; do
    update_file "$file"
    echo ""
done

if [ "$DRY_RUN" = false ]; then
    # 验证更新结果
    print_info "验证更新结果..."
    echo ""

    for file in "${FILES_TO_UPDATE[@]}"; do
        if [ -f "$file" ]; then
            echo "📄 $file:"
            for pattern in "${PATTERNS[@]}"; do
                local new_pattern="${pattern/_${OLD_CHUNK}/_${NEW_CHUNK}}"
                local count=$(grep -c "$new_pattern" "$file" 2>/dev/null || echo "0")
                if [ "$count" -gt 0 ]; then
                    echo "  ✓ $new_pattern (${count}处)"
                fi
            done
            
            # 检查是否还有旧的chunk号
            local old_remaining=0
            for pattern in "${PATTERNS[@]}"; do
                local old_count=$(grep -c "$pattern" "$file" 2>/dev/null || echo "0")
                old_remaining=$((old_remaining + old_count))
            done
            
            if [ "$old_remaining" -gt 0 ]; then
                print_warning "  ⚠️  仍有 ${old_remaining} 处旧的chunk_${OLD_CHUNK}未替换"
            fi
        fi
        echo ""
    done
fi

# 最终结果
echo "========================================"
if [ "$DRY_RUN" = true ]; then
    print_success "预览完成！"
    echo ""
    echo "📋 预览摘要:"
    echo "   旧chunk: ${OLD_CHUNK}"
    echo "   新chunk: ${NEW_CHUNK}"
    echo "   模式: 预览模式 (未实际修改文件)"
    echo ""
    echo "🚀 如果预览结果正确，请运行:"
    echo "   ./update_chunk.sh ${OLD_CHUNK} ${NEW_CHUNK}"
    echo ""
    print_info "预览显示的所有替换都是安全的，不会影响其他数值参数"
else
    print_success "更新完成！"
    echo ""
    echo "📋 摘要:"
    echo "   旧chunk: ${OLD_CHUNK}"
    echo "   新chunk: ${NEW_CHUNK}"
    if [ "$BACKUP_CREATED" = true ]; then
        echo "   备份目录: $BACKUP_DIR"
    fi
    echo ""
    echo "🚀 下一步操作:"
    echo "   sbatch leo.sh"
    echo ""
    
    # 提示检查文件内容
    print_info "建议您检查一下更新后的文件内容是否正确"
    echo "   可以运行: grep -n \"chunk_${NEW_CHUNK}\" leo.sh scripts/gen_vllm_ray_visual.sh"
fi 